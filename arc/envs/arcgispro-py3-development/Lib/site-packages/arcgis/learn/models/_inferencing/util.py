import torch
import numpy as np
from torch import tensor
import torch
import math
from .._unet_utils import is_contiguous as is_cont

def A(*a): return np.array(a[0]) if len(a)==1 else [np.array(o) for o in a]

imagenet_stats = A([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

mean = 255* np.array(imagenet_stats[0], dtype=np.float32)
std  = 255* np.array(imagenet_stats[1], dtype=np.float32)

norm = lambda x: (x-mean)/ std
denorm = lambda x: x * std + mean

def scale_batch(image_batch, model_info, normalization_stats=None, break_extract_bands=False):
    if normalization_stats is None:
        normalization_stats = model_info.get("NormalizationStats", None)
    if break_extract_bands:
        # Only for change detection
        # if subset of extract bands are specified fix this.
        n_bands = len(model_info['ExtractBands']) // 2
        band_min_values = np.array(normalization_stats["band_min_values"])[model_info['ExtractBands'][:n_bands]].reshape(1, -1, 1, 1)
        band_max_values = np.array(normalization_stats["band_max_values"])[model_info['ExtractBands'][:n_bands]].reshape(1, -1, 1, 1)
    else:
        band_min_values = np.array(normalization_stats["band_min_values"])[model_info['ExtractBands']].reshape(1, -1, 1, 1)
        band_max_values = np.array(normalization_stats["band_max_values"])[model_info['ExtractBands']].reshape(1, -1, 1, 1)
    img_scaled = ( image_batch - band_min_values ) / ( band_max_values - band_min_values)
    return img_scaled

def normalize_batch(image_batch, model_info=None, normalization_stats=None):
    if normalization_stats is None:
        normalization_stats = model_info.get("NormalizationStats", None)
    scaled_mean_values = np.array(normalization_stats["scaled_mean_values"])[model_info['ExtractBands']].reshape(1, -1, 1, 1)
    scaled_std_values = np.array(normalization_stats["scaled_std_values"])[model_info['ExtractBands']].reshape(1, -1, 1, 1)
    img_scaled = scale_batch(image_batch, model_info)
    img_normed = ( img_scaled - scaled_mean_values ) / scaled_std_values
    return img_normed

def pred2dict(bb_np, score, cat_str, c):
    # convert to top left x,y bottom right x,y
    return {"x1": bb_np[1],
            "x2": bb_np[3],
            "y1": bb_np[0],
            "y2": bb_np[2],
            "score": score,
            "category": cat_str,
            "class": c}

def to_np(x):
    return x.cpu().numpy()

def load_weights(m, p):
    if p.split('.')[-1] == 'h5':
        sd = torch.load(p, map_location=lambda storage, loc: storage)
    elif p.split('.')[-1] == 'pth':
        # sd = torch.load(p)['model']
        sd = torch.load(p, map_location=lambda storage, loc: storage)['model']
    m.load_state_dict(sd)
    return m

def predict_(model, images, device):
    model = model.to(device)
    images = tensor(images).to(device).float()
    clas, bbox = model(images)
    return clas, bbox
    
def hw2corners(ctr, hw): 
        return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)
    
def actn_to_bb(actn, anchors, grid_sizes):
        actn_bbs = torch.tanh(actn)
        actn_centers = (actn_bbs[:,:2]/2 * grid_sizes) + anchors[:,:2]
        actn_hw = (actn_bbs[:,2:]/2+1) * anchors[:,2:]
        return hw2corners(actn_centers, actn_hw)
    
def nms(boxes, scores, overlap=0.5, top_k=100):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0: return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        keep[count] = i
        count += 1
        if idx.size(0) == 1: break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

def get_nms_preds(b_clas, b_bb, idx, anchors, grid_sizes, classes, nms_overlap, thres):
    
    a_ic = actn_to_bb(b_bb[idx], anchors, grid_sizes)
    clas_pr, clas_ids = b_clas[idx].max(1)
    clas_pr = clas_pr.sigmoid()

    conf_scores = b_clas[idx].sigmoid().t().data

    out1, out2, cc = [], [], []
    for cl in range(1, len(conf_scores)):
        c_mask = conf_scores[cl] > thres
        if c_mask.sum() == 0: continue
        scores = conf_scores[cl][c_mask]
        l_mask = c_mask.unsqueeze(1).expand_as(a_ic)
        boxes = a_ic[l_mask].view(-1, 4)
        ids, count = nms(boxes.data, scores, nms_overlap, 50) # FIX- NMS overlap hardcoded
        ids = ids[:count]
        out1.append(scores[ids])
        out2.append(boxes.data[ids])
        cc.append([cl]*count)
    if cc == []:
        cc = [[0]]
    cc = tensor(np.concatenate(cc))
    if out1 == []:
        out1 = [torch.Tensor()]
    out1 = torch.cat(out1)
    if out2 == []:
        out2 = [torch.Tensor()]
    out2 = torch.cat(out2)
    bbox, clas, prs, thresh = out2, cc, out1, thres  # FIX- hardcoded threshold
    return predictions(bbox,
         to_np(clas), to_np(prs) if prs is not None else None, thresh, classes)

def predictions(bbox, clas=None, prs=None, thresh=0.3, classes=None): # FIX- take threshold from user
    #bb = [bb_hw(o) for o in bbox.reshape(-1,4)]
    if len(classes) is None:
        raise Exception("Classes are None")
    else:
        classes = ['bg'] + classes
    bb = bbox
    if prs is None:  prs  = [None]*len(bb)
    if clas is None: clas = [None]*len(bb)
    predictions = []
    for i, (b, c, pr) in enumerate(zip(bb, clas, prs)):
        if((b[2]>0) and (pr is None or pr > thresh)):
            cat_str = classes[c]
            score = pr * 100
            bb_np = to_np(b).astype('float64')
            predictions.append(pred2dict(bb_np, score, cat_str, c))
    return predictions

def detect_objects_image_space(model, tiles, anchors, grid_sizes, device, classes, nms_overlap, thres, model_info):
    tile_height, tile_width = tiles.shape[2], tiles.shape[3]
    if "NormalizationStats" in model_info:
        img_normed = normalize_batch(tiles, model_info)
    else:
        img_normed = norm(tiles.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)

    clas, bbox = predict_(model, img_normed, device)

    preds = { }

    for batch_idx in range(bbox.size()[0]):
        preds[batch_idx] = get_nms_preds(clas, bbox, batch_idx, anchors, grid_sizes, classes, nms_overlap, thres)

    batch_size = bbox.size()[0]
    side = math.sqrt(batch_size)

    num_boxes = 0
    for batch_idx in range(batch_size):
        num_boxes = num_boxes + len(preds[batch_idx])

    bounding_boxes = np.empty(shape=(num_boxes, 4), dtype=np.float)
    scores = np.empty(shape=(num_boxes), dtype=np.float)
    classes = np.empty(shape=(num_boxes), dtype=np.uint8)

    idx = 0
    for batch_idx in range(batch_size):
        i, j = batch_idx//side, batch_idx%side

        for pred in preds[batch_idx]:
            bounding_boxes[idx, 0] = (pred['y1'] + i)*tile_height
            bounding_boxes[idx, 1] = (pred['x1'] + j)*tile_width
            bounding_boxes[idx, 2] = (pred['y2'] + i)*tile_height
            bounding_boxes[idx, 3] = (pred['x2'] + j)*tile_width
            scores[idx] = pred['score']
            classes[idx] = pred['class']
            idx = idx+1

    return bounding_boxes, scores, classes

def segment_image(model, images, device, predict_bg, model_info):
    model = model.to(device)
    normed_batch_tensor = tensor(images).to(device).float()
    output = model(normed_batch_tensor)
    ignore_mapped_class = model_info.get('ignore_mapped_class', [])
    for k in ignore_mapped_class:
        output[:, k] = -1
    if predict_bg:
        return output.max(dim=1)[1]
    else:
        output[:, 0] = -1
        return output.max(dim=1)[1]

def superres_image(model, images, device):
    model = model.to(device)
    normed_batch_tensor = tensor(images).to(device).float()
    output = model(normed_batch_tensor)
    return output

def cyclegan_image(model, images, device, direction):
    model = model.to(device)
    normed_batch_tensor = tensor(images).to(device).float()
    if direction == 'BtoA':
        output = model.G_A(normed_batch_tensor)
    else:
        output = model.G_B(normed_batch_tensor)
    return output

def pix2pix_image(model, images, device):
    model = model.to(device)
    normed_batch_tensor = tensor(images).to(device).float()
    output = model.G(normed_batch_tensor)
    return output 
    
def remap(tensor, idx2pixel):
    modified_tensor = torch.zeros_like(tensor)
    for id, pixel in idx2pixel.items():
        modified_tensor[tensor == id] = pixel
    return modified_tensor    

def pixel_classify_image(model, tiles, device, classes, predict_bg, model_info):
    class_values = [clas['Value'] for clas in model_info['Classes']]
    is_contiguous = is_cont([0] + class_values)

    if not is_contiguous:
        pixel_mapping = [0] + class_values
        idx2pixel = {i: d for i, d in enumerate(pixel_mapping)}

    tile_height, tile_width = tiles.shape[2], tiles.shape[3]
    if "NormalizationStats" in model_info:
        img_normed = normalize_batch(tiles, model_info)
    else:
        img_normed = norm(tiles.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    semantic_predictions = segment_image(model, img_normed, device, predict_bg, model_info)
    if not is_contiguous:
        semantic_predictions = remap(semantic_predictions, idx2pixel)       
    return semantic_predictions

def pixel_classify_superres_image(model, tiles, device):
    tile_height, tile_width = tiles.shape[2], tiles.shape[3]
    img_normed = norm(tiles.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    superres_predictions = superres_image(model, img_normed, device)
    superres_predictions = (superres_predictions * torch.tensor(imagenet_stats[1]).view(1, -1, 1, 1).to(superres_predictions)) + torch.tensor(imagenet_stats[0]).view(1, -1, 1, 1).to(superres_predictions)
    superres_predictions = superres_predictions.clamp(0, 1)
    return superres_predictions

def pixel_classify_cyclegan_image(model, tiles, device, direction, model_info):
    tile_height, tile_width = tiles.shape[2], tiles.shape[3]
    num_channel = model_info.get("n_channel", None)
    if model_info.get('IsMultispectral', False):
        if direction == 'BtoA':
            norm_stats = model_info.get("NormalizationStats_b", None)
        else:
            norm_stats = model_info.get("NormalizationStats", None)
        img_scaled = scale_batch(tiles, model_info, norm_stats)
        img_normed = -1 + 2*img_scaled
        if img_normed.shape[1] < num_channel:
            cont = []
            for j in range(img_normed.shape[0]):
                tile = img_normed[j,:,:,:]
                last_tile = np.expand_dims(tile[tile.shape[0]-1,:,:], 0)
                res = abs(num_channel - tile.shape[0])
                for i in range(res):
                    tile = np.concatenate((tile, last_tile), axis=0)
                cont.append(tile)
            img_normed = np.stack(cont, axis = 0)
    else:
        img_normed = -1 + 2*tiles
    cyclegan_predictions = cyclegan_image(model, img_normed, device, direction)
    cyclegan_predictions = cyclegan_predictions/2 + 0.5
    return cyclegan_predictions

def pixel_classify_pix2pix_image(model, tiles, device, model_info):
    tile_height, tile_width = tiles.shape[2], tiles.shape[3]
    
    num_chanel = model_info.get("n_channel", None)

    if model_info.get('IsMultispectral', False):
        norm_stats = model_info.get("NormalizationStats", None)
        img_scaled = scale_batch(tiles, model_info, norm_stats)
        img_normed = -1 + 2*img_scaled
        if img_normed.shape[1] < num_chanel:
            cont = []
            for j in range(img_normed.shape[0]):
                tile = img_normed[j,:,:,:]
                last_tile = np.expand_dims(tile[tile.shape[0]-1,:,:], 0)
                res = abs(num_chanel - tile.shape[0])
                for i in range(res):
                    tile = np.concatenate((tile, last_tile), axis=0)
                cont.append(tile)
            img_normed = np.stack(cont, axis = 0)
    else:
        img_normed = norm(tiles.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)

    pix2pix_predictions = pix2pix_image(model, img_normed, device)
    if model_info.get('IsMultispectral', False):
        pix2pix_predictions = pix2pix_predictions/2 + 0.5
    else:
        pix2pix_predictions = (pix2pix_predictions * torch.tensor(imagenet_stats[1]).view(1, -1, 1, 1).to(pix2pix_predictions)) + torch.tensor(imagenet_stats[0]).view(1, -1, 1, 1).to(pix2pix_predictions)
        pix2pix_predictions = pix2pix_predictions.clamp(0, 1)
    
    return pix2pix_predictions

def variable_tile_size_check(json_info, parameters):
    if json_info.get("SupportsVariableTileSize", False):
        parameters.extend(
            [
                {
                    'name': 'tile_size',
                    'dataType': 'numeric',
                    'value': int(json_info['ImageHeight']),
                    'required': False,
                    'displayName': 'Tile Size',
                    'description': 'Tile size used for inferencing'
                }
            ]
        )
    return parameters

def detect_change(model,
                  batch,
                  device,
                  model_info):
    mean = 255 * np.array([0.5] * (len(model_info['ExtractBands']) // 2), dtype=np.float32)
    std = 255 * np.array([0.5] * (len(model_info['ExtractBands']) // 2), dtype=np.float32)
    norm = lambda x: (x-mean) / std
    B, C, H, W = batch.shape
    batch_before = batch[:, :C//2]
    batch_after = batch[:, C//2:]
    
    if "NormalizationStats" in model_info:
        mean = np.array([0.5] * (len(model_info['ExtractBands']) // 2), dtype=np.float32)
        std = np.array([0.5] * (len(model_info['ExtractBands']) // 2), dtype=np.float32)        
        batch_before = scale_batch(batch_before, model_info, break_extract_bands=True)
        batch_after = scale_batch(batch_after, model_info, break_extract_bands=True)        

    batch_before = norm(batch_before.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    batch_after = norm(batch_after.transpose(0, 2, 3, 1)).transpose(0, 3, 1, 2)
    batch_before = torch.tensor(batch_before, device=device).float()
    batch_after = torch.tensor(batch_after, device=device).float()
    from ..._utils.change_detection_data import post_process
    with torch.no_grad():
        predictions = post_process(model(batch_before, batch_after))
    # find the non zero class of the two classes.
    change_class = [c['Value'] for c in model_info['Classes'] if c['Value'] != 0][0]
    predictions[predictions != 0] = change_class
    return predictions[:, 0]

