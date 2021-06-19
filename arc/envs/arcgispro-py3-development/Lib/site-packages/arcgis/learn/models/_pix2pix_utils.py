#Code from https://github.com/fastai/course-v3/blob/master/nbs/dl2/cyclegan_ws.ipynb & https://github.com/eriklindernoren/PyTorch-GAN
import torch
from fastai.vision import nn, Callable, List, LearnerCallback, optim, ifnone, F, flatten_model, requires_grad, SmoothenValue, add_metrics
from .._utils.cyclegan import calculate_activation_statistics, calculate_frechet_distance
from fastprogress.fastprogress import progress_bar
from .._utils.superres import psnr, ssim

import torch.nn as nn
import torch.nn.functional as F
import torch

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


##############################
#           U-NET
##############################


class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(GeneratorUNet, self).__init__()

        self.down1 = UNetDown(in_channels, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        self.down5 = UNetDown(512, 512, dropout=0.5)
        self.down6 = UNetDown(512, 512, dropout=0.5)
        self.down7 = UNetDown(512, 512, dropout=0.5)
        self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        self.up1 = UNetUp(512, 512, dropout=0.5)
        self.up2 = UNetUp(1024, 512, dropout=0.5)
        self.up3 = UNetUp(1024, 512, dropout=0.5)
        self.up4 = UNetUp(1024, 512, dropout=0.5)
        self.up5 = UNetUp(1024, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, out_channels, 4, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)


##############################
#        Discriminator
##############################


class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)

class pix2pix(nn.Module):
    
    def __init__(self, ch_in:int, ch_out:int):
        super().__init__()
        
        self.D = Discriminator(ch_in)
        self.G = GeneratorUNet(ch_in, ch_out)
        self.arcgis_results = False
        
    def forward(self, real_A, real_B):
        fake_B = self.G(real_A)
        if self.arcgis_results: return torch.cat([fake_B[:,None],fake_B[:,None]], 1)
        #if not self.training: 
        return [fake_B]

class AdaptiveLoss(nn.Module):
    def __init__(self, crit):
        super().__init__()
        self.crit = crit
    
    def forward(self, output, target:bool, **kwargs):
        targ = output.new_ones(*output.size()) if target else output.new_zeros(*output.size())
        return self.crit(output, targ, **kwargs)

class Adaptivel1Loss(nn.Module):
    def __init__(self, crit1):
        super().__init__()
        self.crit1 = crit1
    
    def forward(self, output, target:bool, **kwargs):
        targ = output.new_ones(*output.size()) if target else output.new_zeros(*output.size())
        return self.crit1(output, targ, **kwargs)

class pix2pixLoss(nn.Module):
    
    def __init__(self, cgan:nn.Module, lambda_A:float=100., lambda_B:float=100, lambda_idt:float=0.5, lsgan:bool=False):
        super().__init__()
        self.cgan,self.l_A,self.l_B,self.l_idt = cgan,lambda_A,lambda_B,lambda_idt
        self.crit = AdaptiveLoss(F.binary_cross_entropy_with_logits)
        self.crit1 = Adaptivel1Loss(F.mse_loss)
    
    def set_input(self, input):
        self.real_A,self.real_B = input

    def forward(self, output, target):
        fake_B = output[0]
        
        self.gen_loss = self.crit(self.cgan.D(fake_B, self.real_A), True)
        self.l1_loss = torch.mean(F.l1_loss(fake_B, self.real_B))

        return self.gen_loss + (self.l_A * self.l1_loss)

class pix2pixTrainer(LearnerCallback):
    _order = -20 #Need to run before the Recorder
    def _set_trainable(self, D=False):
        gen = (not D)
        requires_grad(self.learn.model.G, gen)
        requires_grad(self.learn.model.D, D)
        if not gen:
            self.opt_D.lr, self.opt_D.mom = self.learn.opt.lr, self.learn.opt.mom
            self.opt_D.wd, self.opt_D.beta = self.learn.opt.wd, self.learn.opt.beta
    
    def on_train_begin(self, **kwargs):
        self.G = self.learn.model.G
        self.D = self.learn.model.D
        self.crit = self.learn.loss_func.crit
        self.crit1 = self.learn.loss_func.crit1

        if not getattr(self,'opt_G',None):
            self.opt_G = self.learn.opt.new([nn.Sequential(*flatten_model(self.G))])
        else: 
            self.opt_G.lr,self.opt_G.wd = self.opt.lr,self.opt.wd
            self.opt_G.mom,self.opt_G.beta = self.opt.mom,self.opt.beta

        if not getattr(self,'opt_D',None):
            self.opt_D = self.learn.opt.new([nn.Sequential(*flatten_model(self.D))])

        self.learn.opt.opt = self.opt_G.opt
        self._set_trainable()
        self.gen_smter,self.l1_smter = SmoothenValue(0.98),SmoothenValue(0.98)
        self.d_smter = SmoothenValue(0.98)
        self.recorder.add_metric_names(['gen_loss', 'l1_loss', 'D_loss'])
        
    def on_batch_begin(self, last_input, **kwargs):
        self.learn.loss_func.set_input(last_input)
    
    def on_backward_begin(self, **kwargs):
        self.l1_smter.add_value(self.loss_func.l1_loss.detach().cpu())
        self.gen_smter.add_value(self.loss_func.gen_loss.detach().cpu())
    
    def on_batch_end(self, last_input, last_output, **kwargs):
        self.G.zero_grad()
        fake_B = last_output[0].detach()
        real_A, real_B = last_input

        self._set_trainable(D=True)

        self.D.zero_grad()

        loss_D = 0.5 * (torch.mean(self.crit1(self.D(real_A, real_B), True)) + torch.mean(self.crit1(self.D(fake_B,real_A), False)))

        self.d_smter.add_value(loss_D.detach().cpu())
        if self.learn.model.training == True:
            loss_D.backward()

        self.opt_D.step()

        self._set_trainable()
        
    def on_epoch_end(self, last_metrics, **kwargs):
        return add_metrics(last_metrics, [s.smooth for s in [self.gen_smter,self.l1_smter,self.d_smter]])

def compute_fid_metric(model, data):
    input_a = []
    input_b = []
    pred_b = []
    for input, target in data.valid_dl:
        input_a.append(input[0]/2+0.5)
        input_b.append(input[1]/2+0.5)
        pred = model.learn.pred_batch(batch=(input,target))
        if isinstance(pred, list):
            pred = pred[0]
        else:
            pred = pred[:,0,:,:,:]
        pred_b.append(pred/2+0.5)
    
    data_len = len(data.valid_ds)
    batch_size = data.batch_size

    m1_b, s1_b = calculate_activation_statistics(batch_size, data_len, input_b)
    m2_b, s2_b = calculate_activation_statistics(batch_size, data_len, pred_b)

    fid_value = calculate_frechet_distance(m1_b, s1_b, m2_b, s2_b)

    return fid_value

def compute_metrics(model, dl, show_progress):
    avg_psnr = 0
    avg_ssim = 0
    model.learn.model.eval()
    with torch.no_grad():
        for input, target in progress_bar(dl, display=False):
            prediction = model.learn.model(input[0], input[1])
            if isinstance(prediction, list):
                prediction = prediction[0]
            else:
                prediction = prediction[:,0,:,:,:]
            avg_psnr += psnr(prediction, input[1])
            avg_ssim += ssim(prediction, input[1])
    return avg_psnr/len(dl), avg_ssim.item()/len(dl)