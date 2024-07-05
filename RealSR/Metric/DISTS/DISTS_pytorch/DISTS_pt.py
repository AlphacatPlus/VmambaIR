# This is a pytoch implementation of DISTS metric.
# Requirements: python >= 3.6, pytorch >= 1.0

import numpy as np
import os,sys
import torch
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F

class L2pooling(nn.Module):
    def __init__(self, filter_size=5, stride=2, channels=None, pad_off=0):
        super(L2pooling, self).__init__()
        self.padding = (filter_size - 2 )//2
        self.stride = stride
        self.channels = channels
        a = np.hanning(filter_size)[1:-1]
        g = torch.Tensor(a[:,None]*a[None,:])
        g = g/torch.sum(g)
        self.register_buffer('filter', g[None,None,:,:].repeat((self.channels,1,1,1)))

    def forward(self, input):
        input = input**2
        out = F.conv2d(input, self.filter, stride=self.stride, padding=self.padding, groups=input.shape[1])
        return (out+1e-12).sqrt()

class DISTS(torch.nn.Module):
    def __init__(self, load_weights=True):
        super(DISTS, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0,4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        self.stage2.add_module(str(4), L2pooling(channels=64))
        for x in range(5, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        self.stage3.add_module(str(9), L2pooling(channels=128))
        for x in range(10, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        self.stage4.add_module(str(16), L2pooling(channels=256))
        for x in range(17, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        self.stage5.add_module(str(23), L2pooling(channels=512))
        for x in range(24, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])
    
        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1,-1,1,1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1,-1,1,1))

        self.chns = [3,64,128,256,512,512]
        self.register_parameter("alpha", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.register_parameter("beta", nn.Parameter(torch.randn(1, sum(self.chns),1,1)))
        self.alpha.data.normal_(0.1,0.01)
        self.beta.data.normal_(0.1,0.01)
        if load_weights:
            # weights = torch.load(os.path.join(sys.prefix, 'weights.pt'))
            weights = torch.load('scripts/metrics/DISTS/DISTS_pytorch/weights.pt')
            self.alpha.data = weights['alpha']
            self.beta.data = weights['beta']
        
    def forward_once(self, x):
        h = (x-self.mean)/self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        return [x,h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]

    def forward(self, x, y, require_grad=False, batch_average=False):
        if require_grad:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)   
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y) 
        dist1 = 0 
        dist2 = 0 
        c1 = 1e-6
        c2 = 1e-6
        w_sum = self.alpha.sum() + self.beta.sum()
        alpha = torch.split(self.alpha/w_sum, self.chns, dim=1)
        beta = torch.split(self.beta/w_sum, self.chns, dim=1)
        for k in range(len(self.chns)):
            x_mean = feats0[k].mean([2,3], keepdim=True)
            y_mean = feats1[k].mean([2,3], keepdim=True)
            S1 = (2*x_mean*y_mean+c1)/(x_mean**2+y_mean**2+c1)
            dist1 = dist1+(alpha[k]*S1).sum(1,keepdim=True)

            x_var = ((feats0[k]-x_mean)**2).mean([2,3], keepdim=True)
            y_var = ((feats1[k]-y_mean)**2).mean([2,3], keepdim=True)
            xy_cov = (feats0[k]*feats1[k]).mean([2,3],keepdim=True) - x_mean*y_mean
            S2 = (2*xy_cov+c2)/(x_var+y_var+c2)
            dist2 = dist2+(beta[k]*S2).sum(1,keepdim=True)

        score = 1 - (dist1+dist2).squeeze()
        if batch_average:
            return score.mean()
        else:
            return score

def prepare_image(image, resize=True):
    if resize and min(image.size) > 256:
        image = transforms.functional.resize(image, 256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0)

if __name__ == '__main__':

    from PIL import Image
    import glob

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # others
    # data_root = '/data1/liangjie/BasicSR_ALL/results/'
    # ref_root = '/data1/liangjie/BasicSR_ALL/datasets/'
    # ref_dirs = ['SISR_Test_matlab/Set5mod12', 'SISR_Test_matlab/Set14mod12', 'SISR_Test_matlab/Manga109mod12', 'SISR_Test_matlab/BSDS100mod12', 'SISR_Test_matlab/General100mod12', 'SISR_Test/Urban100', 'DIV2K/DIV2K_valid_HR/']
    # datasets = ['Set5', 'Set14', 'Manga109', 'BSDS100', 'General100', 'Urban100', 'DIV2K100']
    # img_dirs = ['SRGAN_official', 'ESRGAN_official', 'NatSR_official', 'USRGAN_official', 'SPSR_official', 'SPSR_DF2K', 'ESRGAN_ours_DIV2K', 'ESRGAN_ours_DIV2K_ema', 'ESRGAN_ours_DF2K', 'ESRGAN_ours_DF2K_ema']

    # SFTGAN
    # data_root = '/data1/liangjie/BasicSR_ALL/results'
    # ref_root = '/data1/liangjie/BasicSR_ALL/results/SFTGAN_official'
    # ref_dirs = ['GT'] * 7
    # datasets = ['Set5', 'Set14', 'Manga109', 'BSDS100', 'General100', 'Urban100', 'DIV2K100']
    # img_dirs = ['SFTGAN_official']

    # new
    data_root = 'results/'
    ref_root = 'datasets/'
    ref_dirs = ['DIV2K/DIV2K_valid_HR/']
    datasets = ['DIV2K100']
    img_dirs = ['ESRGAN_ours_DISTS_300k/visualization/']

    logoverall_path = 'results/table_logs/' + 'DISTS_orisize_DISTStrain225k.txt'

    for index in range(len(ref_dirs)):
        ref_dir = os.path.join(ref_root, ref_dirs[index])
        for method in img_dirs:
            img_dir = os.path.join(data_root, method, datasets[index])

            img_list = sorted(glob.glob(os.path.join(img_dir, '*')))

            log_path = 'results/table_logs/' + img_dir.replace('/', '_') + '_DISTS_orisize.txt'

            DISTS_all = []

            for i, img_path in enumerate(img_list):
                file_name = img_path.split('/')[-1]
                if 'DIV2K100' in img_dir and 'SFTGAN' not in img_dir:
                    gt_path = os.path.join(ref_dir, file_name[:4] + '.png')
                elif 'Urban100' in img_dir and 'SFTGAN' not in img_dir:
                    gt_path = os.path.join(ref_dir, file_name[:7] + '.png')
                elif 'SFTGAN' in img_dir:
                    gt_path = os.path.join(ref_dir, file_name.split('_')[0] + '_gt.png')
                    if 'Urban100' in img_dir:
                        gt_path = os.path.join(ref_dir, file_name.split('_')[0] + '_' + file_name.split('_')[1] + '_gt.png')
                else:
                    if '_' in file_name:
                        gt_path = os.path.join(ref_dir, file_name.split('_')[0] + '.png')
                    else:
                        gt_path = os.path.join(ref_dir, file_name)

                ref = prepare_image(Image.open(gt_path).convert("RGB"), resize=False)
                dist = prepare_image(Image.open(img_path).convert("RGB"), resize=False)
                assert ref.shape == dist.shape
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = DISTS().to(device)
                ref = ref.to(device)
                dist = dist.to(device)
                score = model(ref, dist)
                DISTS_all.append(score.item())
                log = f'{i + 1:3d}: {file_name:25}. \tDISTS: {score.item():.6f}.'
                with open(log_path, 'a') as f:
                    f.write(log + '\n')
                # print(log)

            log = f'Average: DISTS: {sum(DISTS_all) / len(DISTS_all):.6f}'
            with open(log_path, 'a') as f:
                f.write(log + '\n')
            log_overall = method + '__' + datasets[index] + '__' + log
            with open(logoverall_path, 'a') as f:
                f.write(log_overall + '\n')
            print(log_overall)

