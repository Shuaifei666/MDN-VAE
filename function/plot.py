import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import numpy as np
from torchvision.utils import save_image
import matplotlib.pyplot as plt
dat_mat = loadmat('./data/HCP_subcortical_CMData_desikan.mat')
tensor = dat_mat['loaded_tensor_sub']
A_mat = np.mean(np.squeeze(tensor[:,:,1,:]), axis=2)
A_mat = A_mat + A_mat.transpose()
for i in range(10):
    data_matrix = tensor[:, :, 1, i]  
    data_matrix = data_matrix + data_matrix.transpose()
    plt.imshow(data_matrix)  
    plt.colorbar()
    plt.title(f"Sample {i+1}")
    plt.savefig(f'results/sample_{i+1}.png')
    plt.clf()  