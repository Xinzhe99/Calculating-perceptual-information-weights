import cv2
import torch
import torch.nn as nn
import math
from torchvision.models import vgg16_bn
import numpy as np
from torchvision import transforms

def get_feature_module(layer_index,device=None):
    vgg = vgg16_bn(pretrained=True, progress=True).features
    vgg.eval()
    # 冻结参数
    for parm in vgg.parameters():
        parm.requires_grad = False
    feature_module = vgg[0:layer_index + 1]
    feature_module.to(device)
    return feature_module

def vgg16_inf(feature_module,img):
    out=feature_module(img).squeeze().permute(1,2,0).detach().cpu().numpy()
    out=(out*255).astype(np.uint8)
    H=out.shape[0]
    W = out.shape[1]
    C = out.shape[2]
    inf=0
    for i in range(out.shape[-1]):
        pic1_lap=cv2.Laplacian(out[:, :, i], -1, ksize=3)
        pic1_norm=np.linalg.norm(pic1_lap)**2
        inf += pic1_norm
    inf=inf/(H*W*C)
    return inf

class PerceptualInfor(nn.Module):
    def __init__(self):
        super(PerceptualInfor, self).__init__()
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.layer_indexs = [5,12,22,32,42]
        self.trans = transforms.Compose([transforms.ToTensor()])
    def forward(self,img1,img2):
        img1 = self.trans(img1).unsqueeze(0).to(self.device)
        img2 = self.trans(img2).unsqueeze(0).to(self.device)
        all_inf1 = 0
        all_inf2 = 0
        c=500#缩放因子,增大会缩小差距
        for index in self.layer_indexs:
            feature_module=get_feature_module(index,self.device)
            all_inf1 +=vgg16_inf(feature_module,img1)
            all_inf2 += vgg16_inf(feature_module,img2)
        all_inf1/=(len(self.layer_indexs)*c)
        all_inf2/=(len(self.layer_indexs)*c)
        w1=math.exp(all_inf1)/(math.exp(all_inf1)+math.exp(all_inf2))
        w2=1-w1
        return w1,w2
#调用方式,在别的py文件中
# import cal_per_weight
# import cv2
img1 = cv2.imread('lytro-01-A.jpg',1)
img2 = cv2.imread('lytro-01-B.jpg',1)
cal_per_weights = cal_per_weight.PerceptualInfor()
w1,w2=cal_per_weights(img1,img2)
print(str(w1)+','+str(w2))