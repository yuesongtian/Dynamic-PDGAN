import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import timm
import numpy as np

from torch_utils import persistence
from pg_modules.blocks import DownBlock, DownBlockPatch, conv2d
import dnnlib

def calc_channels_res(model, inp_res):
    inp = torch.zeros(1, 3, inp_res, inp_res)
    CHANNELS, RESOLUTIONS = [], []

    x = model.patch_embed(inp)
    x = model.pos_drop(x)
    x = model.layers[0](x)
    CHANNELS.append(x.shape[2]); RESOLUTIONS.append(int(x.shape[1]**0.5))
    x = model.layers[1](x)
    CHANNELS.append(x.shape[2]); RESOLUTIONS.append(int(x.shape[1]**0.5))
    x = model.layers[2](x)
    CHANNELS.append(x.shape[2]); RESOLUTIONS.append(int(x.shape[1]**0.5))
    x = model.layers[3](x)
    CHANNELS.append(x.shape[2]); RESOLUTIONS.append(int(x.shape[1]**0.5))

    return CHANNELS, RESOLUTIONS


@persistence.persistent_class  
class SingleDisc(nn.Module):
    def __init__(self, nc=None, ndf=None, start_sz=256, end_sz=8, head=None, separable=False, patch=False):
        super().__init__()
        channel_dict = {4: 512, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64,
                        256: 32, 512: 16, 1024: 8}

        # interpolate for start sz that are not powers of two
        if start_sz not in channel_dict.keys():
            sizes = np.array(list(channel_dict.keys()))
            start_sz = sizes[np.argmin(abs(sizes - start_sz))]
        self.start_sz = start_sz

        # if given ndf, allocate all layers with the same ndf
        if ndf is None:
            nfc = channel_dict
        else:
            nfc = {k: ndf for k, v in channel_dict.items()}

        # for feature map discriminators with nfc not in channel_dict
        # this is the case for the pretrained backbone (midas.pretrained)
        if nc is not None and head is None:
            nfc[start_sz] = nc

        layers = []

        # Head if the initial input is the full modality
        if head:
            layers += [conv2d(nc, nfc[256], 3, 1, 1, bias=False),
                       nn.LeakyReLU(0.2, inplace=True)]

        # Down Blocks
        DB = partial(DownBlockPatch, separable=separable) if patch else partial(DownBlock, separable=separable)
        while start_sz > end_sz:
            layers.append(DB(nfc[start_sz],  nfc[start_sz//2]))
            start_sz = start_sz // 2

        layers.append(conv2d(nfc[end_sz], 1, 4, 1, 0, bias=False))
        self.main = nn.Sequential(*layers)

    def forward(self, x, c):
        return self.main(x)

@persistence.persistent_class  
class MultiScaleD(nn.Module):
    def __init__(
        self,
        channels,
        resolutions,
        ind_blocks=[],    # The number of blocks you want to use. For example, [0,1,2] denotes you want to use 1st, 2nd, 3rd block, from bottom to top.
        proj_type=2,  # 0 = no projection, 1 = cross channel mixing, 2 = cross scale mixing
        cond=0,
        separable=False,
        patch=False,
        **kwargs,
    ):
        super().__init__()

        assert len(ind_blocks) > 0

        # the first disc is on the lowest level of the backbone
        self.disc_in_channels = [channels[ind] for ind in ind_blocks]
        self.disc_in_res = [resolutions[ind] for ind in ind_blocks]
        # I try with one disc, which attaches to the top of the backbone.
        #self.disc_in_channels = channels[-num_discs:]
        #self.disc_in_res = resolutions[-num_discs:]
        Disc = SingleDiscCond if cond else SingleDisc

        mini_discs = []
        for i, (cin, res) in enumerate(zip(self.disc_in_channels, self.disc_in_res)):
            start_sz = res if not patch else 16
            mini_discs += [str(i), Disc(nc=cin, start_sz=start_sz, end_sz=8, separable=separable, patch=patch)],
        self.mini_discs = nn.ModuleDict(mini_discs)

    def forward(self, features, c):
        all_logits = []
        for k, disc in self.mini_discs.items():
            fea = disc(features[k], c)
            all_logits.append(fea.view(fea.size(0), -1))

        all_logits = torch.cat(all_logits, dim=1)
        return all_logits

@persistence.persistent_class  
class Swin(torch.nn.Module):

    def __init__(self, **ext_kwargs):
        super().__init__(
        )

        self.model = timm.create_model('swin_tiny_patch4_window7_224')
        st = torch.load('/apdcephfs/share_1367250/yuesongtian/pretrained_models/pretrained_ViT/swin_tiny_patch4_window7_224.pth', map_location='cpu')
        self.CHANNELS, self.RESOLUTIONS = calc_channels_res(self.model, 224)
        new_st = {}
        for each in st['model'].keys():
            if 'encoder.' in each:
                newk = each.replace('encoder.','')
                new_st[newk] = st['model'][each]
        self.model.load_state_dict(new_st, strict=False)

        self.model.eval()
        self.model.requires_grad = False

        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])      
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward_features_custom_swin(self, x, return_intermediate=False):
        x = self.model.patch_embed(x)
        if self.model.ape:
            x = x + self.model.absolute_pos_embed
        x = self.model.pos_drop(x)
        x = self.model.layers(x)     
        x = self.model.norm(x)
        if return_intermediate:
            return x.transpose(1, 2)
        
        x = self.model.avgpool(x.transpose(1, 2)) 
        x = torch.flatten(x, 1)
        return x

    def forward_features_intermediate(self, x, return_intermediate=False):
        """
        Return the output features of the blocks of Swin Transformer.
        """
        x = self.model.patch_embed(x)
        if self.model.ape:
            x = x + self.model.absolute_pos_embed
        x = self.model.pos_drop(x)
        out1 = self.model.layers[0](x)
        out2 = self.model.layers[1](out1)
        out3 = self.model.layers[2](out2)
        out4 = self.model.layers[3](out3)

        out = {'0': out1.permute(0,2,1).view(-1, self.CHANNELS[0], self.RESOLUTIONS[0], self.RESOLUTIONS[0]),
               '1': out2.permute(0,2,1).view(-1, self.CHANNELS[1], self.RESOLUTIONS[1], self.RESOLUTIONS[1]),
               '2': out3.permute(0,2,1).view(-1, self.CHANNELS[2], self.RESOLUTIONS[2], self.RESOLUTIONS[2]),
               '3': out4.permute(0,2,1).view(-1, self.CHANNELS[3], self.RESOLUTIONS[3], self.RESOLUTIONS[3])}

        return out

    def __call__(self, image):
        #image = F.interpolate(image*0.5+0.5, size=(224,224), mode='area')
        image = F.interpolate(image, size=(224,224), mode='area')
        #image = image - self.image_mean[:, None, None].to(image.device)
        #image /= self.image_std[:, None, None].to(image.device)
            
        return self.forward_features_intermediate(image)

def _make_efficientnet(model):
    pretrained = nn.Module()
    pretrained.layer0 = nn.Sequential(model.conv_stem, model.bn1, model.act1, *model.blocks[0:2])
    pretrained.layer1 = nn.Sequential(*model.blocks[2:3])
    pretrained.layer2 = nn.Sequential(*model.blocks[3:5])
    pretrained.layer3 = nn.Sequential(*model.blocks[5:9])
    return pretrained

@persistence.persistent_class  
class EfficientNet(torch.nn.Module):

    def __init__(self, **ext_kwargs):
        super().__init__(
        )

        self.model = timm.create_model('tf_efficientnet_lite0')
        st = torch.load('/apdcephfs/share_1367250/yuesongtian/pretrained_models/pretrained_ViT/tf_efficientnet_lite0-0aa007d2.pth', map_location='cpu')
        self.model.load_state_dict(st, strict=False)

        self.model.eval()
        self.model.requires_grad = False
        self.fea_ext = _make_efficient(self.model)

        self.image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
        self.image_std = torch.tensor([0.229, 0.224, 0.225])      
        self.pool = nn.AdaptiveAvgPool2d((1,1))

    def forward_input(self, x):
        x = self.fea_ext(x)
        #x = self.model.global_pool(x)
        
        return x

    def __call__(self, image):
        #image = F.interpolate(image*0.5+0.5, size=(224,224), mode='area')
        image = F.interpolate(image, size=(256,256), mode='bilinear')
        #image = image - self.image_mean[:, None, None].to(image.device)
        #image /= self.image_std[:, None, None].to(image.device)
        #image = (image+1.) / 2.0
           
        return self.forward_input(image)
