import torch
from torch import Tensor
from torch.optim import Adam
from torch.optim.lr_scheduler import _LRScheduler, StepLR
from torch.optim.optimizer import Optimizer
from torchaudio.transforms import MelSpectrogram
from torch import nn
import torch.nn.functional as F
import itertools
import math

from sonata.models import BaseModule
from sonata.utils import ParametersCounter

IMG_SHAPE = 256
WAV_SHAPE = 8192

class Block1d(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride=1, bias=False, upsample=False):
        super().__init__()
        self.upsample = upsample

        self.conv = nn.Conv1d(in_features, out_features, kernel, stride=stride, padding=(kernel-1)//2, bias=bias)
        self.norm = nn.BatchNorm1d(out_features)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='linear')
        return self.act(self.norm(self.conv(x)))

class Block2d(nn.Module):
    def __init__(self, in_features, out_features, kernel, stride=1, bias=False, upsample=False):
        super().__init__()
        self.upsample = upsample

        self.conv = nn.Conv2d(in_features, out_features, kernel, stride=stride, padding=(kernel-1)//2, bias=bias)
        self.norm = nn.BatchNorm2d(out_features)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False, recompute_scale_factor=False)
        return self.act(self.norm(self.conv(x)))

class ImgGenerator(nn.Module):
    def __init__(self, wav_shape):
        super().__init__()

        bottleneck_shape = int(math.sqrt(wav_shape / 8))
        
        self.encoder = nn.Sequential(
            Block1d(1, 32, 3, stride=2),
            Block1d(32, 16, 3, stride=2),
            Block1d(16, 16, 3, stride=2),
            Block1d(16, 16, 3, stride=1).conv,
            nn.Unflatten(dim=2, unflattened_size=torch.Size([bottleneck_shape, bottleneck_shape]))
        )

        self.decoder = nn.Sequential(
            Block2d(16, 16, 3, upsample=True),
            Block2d(16, 16, 3, upsample=True),
            Block2d(16, 32, 3, upsample=True),
            Block2d(32, 3, 3).conv,
        )


    def forward(self, x):
        emb = self.encoder(x)
        img = self.decoder(emb)
        img = torch.tanh(img)

        return img

class WavGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            Block2d(3, 32, 3, stride=2),
            Block2d(32, 16, 3, stride=2),
            Block2d(16, 16, 3, stride=2),
            Block2d(16, 16, 3, stride=1).conv,
            nn.Flatten(start_dim=2),
        )

        self.decoder = nn.Sequential(
            Block1d(16, 16, 3, upsample=True),
            Block1d(16, 16, 3, upsample=True),
            Block1d(16, 32, 3, upsample=True),
            Block1d(32, 1, 3).conv,
        )


    def forward(self, x):
        emb = self.encoder(x)
        wav = self.decoder(emb)
        wav = torch.tanh(wav)
        
        print(wav.shape)
        return wav

class ImgDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            Block2d(3, 32, 3, stride=2),
            Block2d(32, 64, 3, stride=2),
            Block2d(64, 128, 3, stride=2),
            Block2d(128, 256, 3, stride=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        self.clf = nn.Sequential(
            nn.Linear(256, 1)
        )


    def forward(self, x):
        emb = self.encoder(x)
        emb = emb.view(x.size(0), -1)
        logits = self.clf(emb)

        return logits

class WavDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            Block1d(1, 32, 3, stride=2),
            Block1d(32, 64, 3, stride=2),
            Block1d(64, 128, 3, stride=2),
            Block1d(128, 256, 3, stride=1),
            nn.AdaptiveAvgPool1d(1)
        )

        self.clf = nn.Sequential(
            nn.Linear(256, 1)
        )


    def forward(self, x):
        emb = self.encoder(x)
        emb = emb.view(x.size(0), -1)
        logits = self.clf(emb)

        return logits

class CycleGANModel(BaseModule):
    def __init__(
            self,
            device=torch.device('cpu'),
            learning_rate=1e-3,
            img_rec_lambda=10,
            wav_rec_lambda=10,
            img_shape=IMG_SHAPE,
            wav_shape=WAV_SHAPE
        ):
        super().__init__()
        self.device = device
        self.img_rec_lambda = img_rec_lambda
        self.wav_rec_lambda = wav_rec_lambda

        self.img2wav_G = WavGenerator().to(device)
        self.wav2img_G = ImgGenerator(wav_shape).to(device)
        self.img2wav_D = WavDiscriminator().to(device)
        self.wav2img_D = ImgDiscriminator().to(device)

        self.criterion_GAN = nn.BCEWithLogitsLoss()
        self.criterion_cycle = nn.L1Loss()

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

    def forward(
            self,
            x
        ):
        img, wav = x

        fake_wav = self.img2wav_G(img)
        fake_img = self.wav2img_G(wav)
        rec_wav = self.img2wav_G(fake_img)
        rec_img = self.wav2img_G(fake_wav)
        
        _ = self.img2wav_D(fake_wav.detach())
        _ = self.wav2img_D(fake_img.detach())

        return fake_wav

    def training_step(
            self,
            batch,
            batch_idx,
            optimizer_idx,
        ):
        
        img, wav = batch
        img = img.to(self.device)
        wav = wav.to(self.device)
        zeros = torch.zeros(img.shape[0]).to(self.device)
        ones = torch.ones(img.shape[0]).to(self.device)

        fake_wav = self.img2wav_G(img)
        fake_img = self.wav2img_G(wav)
        rec_wav = self.img2wav_G(fake_img)
        rec_img = self.wav2img_G(fake_wav)

        self.set_requires_grad([self.img2wav_D, self.wav2img_D], False)
        self.optimizer_G.zero_grad()

        loss_img2wav_G = self.criterion_GAN(self.img2wav_D(fake_wav), ones)
        loss_wav2img_G = self.criterion_GAN(self.wav2img_D(fake_img), ones)
        loss_img_rec_cycle = self.img_rec_lambda * self.criterion_cycle(rec_img, img)
        loss_wav_rec_cycle = self.wav_rec_lambda * self.criterion_cycle(rec_wav, wav)
        
        loss_G = loss_img2wav_G + loss_wav2img_G + loss_img_rec_cycle + loss_wav_rec_cycle

        loss_G.backward()
        self.optimizer_G.step()

        self.set_requires_grad([self.img2wav_D, self.wav2img_D], True)
        self.optimizer_D.zero_grad()

        loss_img2wav_D_fake = self.criterion_GAN(self.img2wav_D(fake_wav.detach()), zeros)
        loss_img2wav_D_real = self.criterion_GAN(self.img2wav_D(wav), ones)
        loss_wav2img_D_fake = self.criterion_GAN(self.img2wav_D(fake_img.detach()), zeros)
        loss_wav2img_D_real = self.criterion_GAN(self.wav2img_D(img), ones)
        
        loss_img2wav_D = (loss_img2wav_D_fake + loss_img2wav_D_real) / 2
        loss_wav2img_D = (loss_wav2img_D_fake + loss_wav2img_D_real) / 2

        loss_D = loss_img2wav_D + loss_wav2img_D

        loss_D.backward()
        self.optimizer_D.step()

        return [loss_G, loss_D]

    def validation_step(
            self,
            batch,
            batch_idx,
        ):

        losses = self.training_step(
            batch=batch,
            batch_idx=batch_idx,
            optimizer_idx=0,
        )

        return losses

    def configure_optimizers(
            self
        ):
        
        self.optimizer_G = Adam(itertools.chain(self.img2wav_G.parameters(), self.wav2img_G.parameters()), lr=self.learning_rate)
        self.optimizer_D = Adam(itertools.chain(self.nimg2wav_D.parameters(), self.wav2img_D.parameters()), lr=self.learning_rate)

        return [self.optimizer_G, self.optimizer_D], []

if __name__ == '__main__':
    
    model = CycleGANModel()
    n_params = ParametersCounter.count(
        model=model,
        trainable=True,
    )
    #pdb.set_trace()
    print(n_params)

    inputs = (torch.randn(4, 3, IMG_SHAPE, IMG_SHAPE), torch.randn(4, 1, WAV_SHAPE))    
    outputs = model(inputs)
    print(outputs.shape)

