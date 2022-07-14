#%%
from stylegan_model import G_mapping
from stylegan_model import G_synthesis

import torch
import torchvision
from collections import OrderedDict
import torch.nn as nn
import numpy as np
from estimator import PCAEstimator


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
    

    # PCA
    n_samples = 10**4
    seed = np.random.randint(np.iinfo(np.int32).max)
    rng = np.random.RandomState(seed)
    x = torch.from_numpy(
                    rng.standard_normal(512 * n_samples)
                    .reshape(n_samples, 512)).float().to(device)
    
    
    w = g_mapping(x) #[10**6, 18, 512]
    w = w[:, 0, :] #[10**6, 512]
    
    K = 512 #주성분 개수
    pca = PCAEstimator(n_components=K)
    pca.fit(w.cpu().detach().numpy())
    pca_comp, _, _ = pca.get_components()
    np.save('pca_comp/pca_ffhq_10_4.npy', pca_comp)
# %%
