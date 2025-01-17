from __future__ import print_function
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
from scipy.linalg import sqrtm

torch.manual_seed(11111)
###Load data set
device = torch.device("cuda")
dat_mat = loadmat('./data/HCP_subcortical_CMData_desikan.mat')
tensor = dat_mat['loaded_tensor_sub']

###Load the adjacentcy matrix
A_mat = np.mean(np.squeeze(tensor[18:86, 18:86,1,:]), axis=2)
A_mat = A_mat + A_mat.transpose()



### Choose the proper type of loss function 
loss_type = "poisson"
### If the loss_type = "poisson", set the proper offset for the mean
offset = 200 
### Set the neighborhood size for GATE
n_size = 32


### Load networks
net_data = []
for i in range(tensor.shape[3]):
    ith = np.float32(tensor[:,:,0,i] + np.transpose(tensor[:,:,0,i]))
    np.fill_diagonal(ith,np.mean(ith, 0))
    ith = ith[18:86, 18:86]
    ith = ith.flatten()
    ith = ith/offset
    net_data.append(ith)

batch_size = 256
tensor_y = torch.stack([torch.Tensor(i) for i in net_data])
y = utils.TensorDataset(tensor_y) # create your datset
train_loader = utils.DataLoader(net_data, batch_size) 
#train_loader = utils.DataLoader(y, batch_size, shuffle=True)



def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)



class GraphCNN(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphCNN, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        self.mask_flag = True 
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight*self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)



class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        latent_dim = 68
        self.fc11 =  nn.Linear(68*68, 1024)
        self.fc12 =  nn.Linear(68*68, 1024)
        self.fc111 = nn.Linear(1024,128)
        self.fc222 = nn.Linear(1024,128)
        self.fc21 = nn.Linear(128, latent_dim)
        self.fc22 = nn.Linear(128, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 68)
        self.fc32 = nn.Linear(latent_dim,68)
        self.fc33 = nn.Linear(latent_dim,68)
        self.fc34 = nn.Linear(latent_dim,68)
        self.fc35 = nn.Linear(latent_dim,68)
        self.fc4 = GraphCNN(68,68)
        self.fc5 = GraphCNN(68,68)
        self.fc6 = GraphCNN(68,68)
        self.fc7 = GraphCNN(68,68)
        self.fc8 = GraphCNN(68,68)
        self.fcintercept = GraphCNN(68*68, 68*68)
    def encode(self, x):
        h11 = F.relu(self.fc11(x))
        h11 = F.relu(self.fc111(h11))
        h12 = F.relu(self.fc12(x))
        h12 = F.relu(self.fc222(h12))
        return self.fc21(h11), self.fc22(h12)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self, z):
        h31= F.sigmoid(self.fc3(z))
        h31= F.sigmoid(self.fc4(h31))
        h31_out = torch.bmm(h31.unsqueeze(2), h31.unsqueeze(1))
        h32 = F.sigmoid(self.fc32(z))
        h32 = F.sigmoid(self.fc5(h32))
        h32_out = torch.bmm(h32.unsqueeze(2), h32.unsqueeze(1))
        h33 = F.sigmoid(self.fc33(z))
        h33 = F.sigmoid(self.fc6(h33))
        h33_out = torch.bmm(h33.unsqueeze(2), h33.unsqueeze(1))
        h34 = F.sigmoid(self.fc34(z))
        h34 = F.sigmoid(self.fc7(h34))
        h34_out = torch.bmm(h34.unsqueeze(2), h34.unsqueeze(1))
        h35 = F.sigmoid(self.fc35(z))
        h35 = F.sigmoid(self.fc8(h35))
        h35_out = torch.bmm(h35.unsqueeze(2), h35.unsqueeze(1))
        h30 = F.sigmoid(h31_out + h32_out + h33_out + h34_out+ h35_out)
        h30 = h30.view(-1, 68*68)
        h30 = self.fcintercept(h30)
        # h30 =torch.exp(h30)
        return h30.view(-1, 68*68), h31+h32+h33+h34
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 68*68))
        z = self.reparameterize(mu, logvar)
        recon, x_latent = self.decode(z)
        return recon.view(-1, 68*68), mu, logvar, x_latent
    def set_mask(self, masks):
        self.fc4.set_mask(masks[0])
        self.fc5.set_mask(masks[1])
        self.fc6.set_mask(masks[2])
        self.fc7.set_mask(masks[3])
        self.fc8.set_mask(masks[4])
        self.fcintercept.set_mask(masks[5])



def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x , reduction='sum')
    BCE = F.poisson_nll_loss(recon_x , x, reduction='sum', log_input=True)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(f"BCE (Poisson NLL): {BCE.item()}, KLD: {KLD.item()}")
    return BCE + KLD



def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, _ = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, (data) in enumerate(train_loader):
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(train_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))




device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)
masks = []
mask_2NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_2NN)).float())
mask_4NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_4NN)).float())
mask_8NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_8NN)).float())
mask_16NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_16NN)).float())
mask_32NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_32NN)).float())
mask_intercept = np.identity(68*68)
masks.append(torch.from_numpy(np.float32(mask_intercept)).float())
model.set_mask(masks)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
batch_size = 256
learning_rate=0.01


for epoch in range(500):
    # if epoch < 5:
    #     model.fc11.requires_grad = False
    #     model.fc21.requires_grad = False
    # if epoch >= 5:
    #     model.fc11.requires_grad = True
    #     model.fc21.requires_grad = True
    train(epoch)


from skimage.metrics import structural_similarity as ssim
net_data_array = np.array(net_data).astype(np.float32)  

batch_size = 64  
data_loader = torch.utils.data.DataLoader(net_data_array, batch_size=batch_size)

generated_matrices = []

model.eval()
with torch.no_grad():
    for batch_data in data_loader:
        batch_data = batch_data.to(device)
        recon_batch, _, _, _ = model(batch_data)
        recon_batch = recon_batch.cpu().numpy()
        for i in range(recon_batch.shape[0]):
            generated_matrix = recon_batch[i].reshape(68, 68)
            generated_matrices.append(generated_matrix)

generated_matrices = np.array(generated_matrices)
print('Generated matrices shape:', generated_matrices.shape)


real_matrices = net_data_array.reshape(-1, 68, 68)
print('Real matrices shape:', real_matrices.shape)


real_matrices = real_matrices.astype(np.float32)
generated_matrices = generated_matrices.astype(np.float32)

max_value = real_matrices.max()
real_matrices /= max_value
generated_matrices /= max_value
if generated_matrices.min() < 0:
    generated_matrices = np.maximum(generated_matrices, 0)

def compute_mse(real_matrices, generated_matrices):
    mse_values = ((real_matrices - generated_matrices) ** 2).mean(axis=(1, 2))
    average_mse = mse_values.mean()
    return average_mse

mse_value = compute_mse(real_matrices, generated_matrices)
print('Average MSE:', mse_value)


def compute_psnr(real_matrices, generated_matrices, max_value=1.0):
    mse_values = ((real_matrices - generated_matrices) ** 2).mean(axis=(1, 2))
    psnr_values = np.where(mse_values == 0, np.inf, 20 * np.log10(max_value) - 10 * np.log10(mse_values))
    average_psnr = psnr_values[np.isfinite(psnr_values)].mean()
    return average_psnr

psnr_value = compute_psnr(real_matrices, generated_matrices, max_value=1.0)
print('Average PSNR:', psnr_value)

def compute_ssim(real_matrices, generated_matrices):
    ssim_values = []
    for real_mat, gen_mat in zip(real_matrices, generated_matrices):
        ssim_value = ssim(real_mat, gen_mat, data_range=1.0)
        ssim_values.append(ssim_value)
    average_ssim = np.mean(ssim_values)
    return average_ssim

average_ssim = compute_ssim(real_matrices, generated_matrices)
print('Average SSIM:', average_ssim)

def compute_spectral_features(matrix, k=None):
    eigenvalues = np.linalg.eigvals(matrix)
    eigenvalues = np.real(eigenvalues)
    eigenvalues = np.sort(eigenvalues)[::-1]
    if k is not None:
        eigenvalues = eigenvalues[:k]
    return eigenvalues

def calculate_fid(mu1, sigma1, mu2, sigma2):
    diff = mu1 - mu2
    covmean, _ = sqrtm(sigma1.dot(sigma2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

k = 30

real_features = []
for matrix in real_matrices:
    features = compute_spectral_features(matrix, k=k)
    real_features.append(features)
real_features = np.array(real_features)

generated_features = []
for matrix in generated_matrices:
    features = compute_spectral_features(matrix, k=k)
    generated_features.append(features)
generated_features = np.array(generated_features)

mu_real = np.mean(real_features, axis=0)
sigma_real = np.cov(real_features, rowvar=False)

mu_gen = np.mean(generated_features, axis=0)
sigma_gen = np.cov(generated_features, rowvar=False)

fid_value = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
print('FID 值:', fid_value)


import numpy as np
import networkx as nx
from scipy.linalg import sqrtm

def compute_graph_features(matrix):
    # 矩阵归一化
    max_value = matrix.max()
    if max_value > 0:
        matrix_normalized = matrix / max_value
    else:
        matrix_normalized = np.zeros_like(matrix)

    # 将连接强度转换为距离
    epsilon = 1e-5
    distance_matrix = 1 / (matrix_normalized + epsilon)

    # 创建加权图
    G = nx.from_numpy_array(matrix_normalized)

    # 计算加权度数
    degrees = np.array([val for (node, val) in G.degree(weight='weight')])
    degree_mean = degrees.mean()
    degree_std = degrees.std()

    # 计算加权聚类系数
    clustering_coeffs = np.array(list(nx.clustering(G, weight='weight').values()))
    clustering_mean = clustering_coeffs.mean()
    clustering_std = clustering_coeffs.std()

    # 添加边的距离属性
    edge_distances = {}
    for i, j in G.edges():
        edge_distances[(i, j)] = distance_matrix[i, j]
    nx.set_edge_attributes(G, edge_distances, 'distance')

    # 计算加权平均最短路径长度
    try:
        avg_shortest_path_length = nx.average_shortest_path_length(G, weight='distance')
    except nx.NetworkXError:
        # 处理不连通图的情况
        lengths = []
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if len(subgraph) > 1:
                length = nx.average_shortest_path_length(subgraph, weight='distance')
                lengths.append(length)
        avg_shortest_path_length = np.mean(lengths) if lengths else 0

    # 特征向量
    features = np.array([
        degree_mean,
        degree_std,
        clustering_mean,
        clustering_std,
        avg_shortest_path_length
    ])
    return features

def calculate_fid(features_real, features_generated):
    # 计算均值和协方差矩阵
    mu_real = np.mean(features_real, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)

    mu_gen = np.mean(features_generated, axis=0)
    sigma_gen = np.cov(features_generated, rowvar=False)

    # 计算均值差的平方
    diff = mu_real - mu_gen
    mean_diff = diff.dot(diff)

    # 计算协方差矩阵的平方根
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    # 处理可能的复数结果
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    # 计算 FID
    fid = mean_diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return fid

# 提取图论特征
features_real = [compute_graph_features(matrix) for matrix in real_matrices]
features_generated = [compute_graph_features(matrix) for matrix in generated_matrices]

# 转换为 NumPy 数组
features_real = np.array(features_real)
features_generated = np.array(features_generated)

# 计算 FID
fid_value = calculate_fid(features_real, features_generated)
print("图论特征方法的 FID 值：", fid_value)
