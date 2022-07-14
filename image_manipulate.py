#%%
from stylegan_model import G_mapping
from stylegan_model import G_synthesis

import torch
from collections import OrderedDict
import torch.nn as nn
import numpy as np

from torchvision.utils import save_image

if __name__=="__main__":
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    resolution = 1024
    weight_file = 'weights/model_trained.pth'

    g_all = nn.Sequential(OrderedDict([
        ('g_mapping', G_mapping()),
        ('g_synthesis', G_synthesis(resolution=resolution))    
    ]))
    g_all.load_state_dict(torch.load(weight_file, map_location=device))
    g_mapping = g_all[0].to(device)
    g_synthesis = g_all[1].to(device)

    pca_comp = np.load('pca_comp/pca_ffhq_10_4.npy')
    V = torch.tensor(pca_comp).transpose(0, 1) #[512, K]

 
    seed = np.random.randint(np.iinfo(np.int32).max)
    rng = np.random.RandomState(seed)
    sample_z = torch.from_numpy(
                    rng.standard_normal(512)
                    .reshape(1, 512)).float().to(device)
    sample_w = g_mapping(sample_z) # [1, 18, 512]
    sample_w = sample_w[:, 0, :] # [1, 512]
    
    K = V.shape[1]
    num_imgs = 7 # 변화 단계 설정
    layer_num = 6 # 첫번쨰 layer부터 몇개의 layer를 수정할것인지 정하기
    for i in range(num_imgs + 1):
        control_params = torch.zeros(K) # control parameter, [K]
        control_params[9] = (1/num_imgs)*i*4 -2
        direction = torch.matmul(V, control_params).reshape(1, -1).to(device) # [1, 512]
        sample_direction_w = sample_w + direction
        sample_direction_w = sample_direction_w.unsqueeze(1).expand(-1, layer_num, -1) # [1, layer_num, 512]
        sample_rest_w = sample_w.unsqueeze(1).expand(-1, 18-layer_num, -1)
        final_w = torch.cat([sample_direction_w, sample_rest_w], dim=1).to(device)
        syn_img = g_synthesis(final_w)
        syn_img = (syn_img+1.0)/2.0
        save_image(syn_img.clamp(0,1),"images/ganspace{}.png".format(i))
# %%
