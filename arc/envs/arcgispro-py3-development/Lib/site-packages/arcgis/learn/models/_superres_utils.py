import PIL
try:
    import torch
    from torchvision.models import vgg16_bn
    import torch.nn.functional as F
    import torch.nn as nn
    from torch.nn import Module as NnModule
    from fastai.vision import resize_to
    from fastai.callbacks import hook_outputs
    from fastai.torch_core import requires_grad, children
    from fastprogress.fastprogress import progress_bar
    from .._utils.superres import psnr, ssim
    HAS_FASTAI = True
except Exception as e:
    #import_exception = "\n".join(traceback.format_exception(type(e), e, e.__traceback__))
    class NnModule():
        pass
    HAS_FASTAI = False


def resize_one(fn, i, path_lr, size, path_hr, img_size):
    dest = path_lr/fn.relative_to(path_hr)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img = PIL.Image.open(fn)
    targ_sz = resize_to(img, size, use_min=True)
    img = img.resize(targ_sz, resample=PIL.Image.BILINEAR).convert('RGB')
    #img = img.resize((img_size,img_size), resample=PIL.Image.BILINEAR).convert('RGB')
    dest = dest.with_suffix(".jpeg")
    img.save(dest)


def gram_matrix(x):
    n,c,h,w = x.size()
    x = x.view(n, c, -1)
    return (x @ x.transpose(1,2))/(c*h*w)

class FeatureLoss(nn.Module):
    def __init__(self, m_feat, layer_ids, layer_wgts, base_loss=F.l1_loss):
        super().__init__()
        self.m_feat = m_feat
        self.base_loss = base_loss
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel',]

    def make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]
    
    def forward(self, input, target):
        out_feat = self.make_features(target, clone=True)
        in_feat = self.make_features(input)
        self.feat_losses = [self.base_loss(input,target)]
        self.feat_losses += [self.base_loss(f_in, f_out)*w
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.feat_losses += [self.base_loss(gram_matrix(f_in), gram_matrix(f_out))*w**2 * 5e3
                             for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)]
        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)
    
    def __del__(self): self.hooks.remove()

def create_loss(device_type='cuda'):
    if device_type=='cuda':
        vgg_m = vgg16_bn(True).features.cuda().eval()
    else:
        vgg_m = vgg16_bn(True).features.eval()
    requires_grad(vgg_m, False)
    blocks = [i-1 for i,o in enumerate(children(vgg_m)) if isinstance(o,nn.MaxPool2d)]
    blocks, [vgg_m[i] for i in blocks]
    feat_loss = FeatureLoss(vgg_m, blocks[2:5], [5,15,2])
    return feat_loss

def compute_metrics(model, dl, show_progress):
    avg_psnr = 0
    avg_ssim = 0
    model.learn.model.eval()
    with torch.no_grad():
        for input, target in progress_bar(dl, display=False):
            prediction = model.learn.model(input)
            avg_psnr += psnr(prediction, target)
            avg_ssim += ssim(prediction, target)
    return avg_psnr/len(dl), avg_ssim.item()/len(dl)

def get_resize(y, z, max_size, f):
    if y*f <= max_size and z*f <= max_size:
        y_new = y*f
        z_new = z*f
    else:
        if y > z:
            y_new = max_size
            z_new = int(round_up_to_even(z * max_size / y))
        else:
            z_new = max_size
            y_new = int(round_up_to_even(y * max_size / z))
    return (y_new, z_new)


