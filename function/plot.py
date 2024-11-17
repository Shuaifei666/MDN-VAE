import numpy as np
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms
from tqdm import tqdm
from scipy.io import loadmat
from torch.autograd import Variable
import numpy as np
import torch.utils.data as utils
from torchvision.utils import save_image
import matplotlib.pyplot as plt
dat_mat = loadmat('./data/HCP_subcortical_CMData_desikan.mat')
tensor = dat_mat['loaded_tensor_sub']
A_mat = np.mean(np.squeeze(tensor[:,:,1,:]), axis=2)
A_mat = A_mat + A_mat.transpose()
for i in range(10):
    # 提取第 i 个样本的矩阵
    data_matrix = tensor[:, :, 1, i]  # 第三维度为 1，样本索引为 i
    # 对矩阵进行对称处理（如果需要）
    data_matrix = data_matrix + data_matrix.transpose()
    # 绘制并保存图像
    plt.imshow(data_matrix)  # 您可以选择其他 colormap，例如 'gray'
    plt.colorbar()
    plt.title(f"Sample {i+1}")
    plt.savefig(f'results/sample_{i+1}.png')
    plt.clf()  # 清除当前图像，准备绘制下一个