from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm
from scipy.io import loadmat
from torch.autograd import Variable
import numpy as np
import torch.utils.data as utils
from torchvision.utils import save_image
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
import networkx as nx
from skimage.metrics import structural_similarity as ssim
import os
import matplotlib.pyplot as plt
import numpy as np
torch.manual_seed(11111)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dat_mat = loadmat('./data/desikan_fc_all.mat')
all_fc_data = dat_mat['all_fc'] 
fc_list = all_fc_data[0]

filtered_fc = [mat for mat in fc_list if mat.shape == (87,87)]

tensor = np.stack(filtered_fc, axis=-1)
print("Tensor shape:", tensor.shape)  # (87,87,N)

A_mat = np.mean(tensor[18:86, 18:86, :], axis=2)

net_data = []
for i in range(tensor.shape[2]):
    ith = np.float32(tensor[:,:,i] + tensor[:,:,i].T)
    np.fill_diagonal(ith, np.mean(ith, 0))
    ith = ith[18:86, 18:86]
    ith = (ith + 1.0) / 2.0
    ith = ith.flatten()
    net_data.append(ith)
net_data = np.array(net_data, dtype=np.float32)

num_samples = len(net_data)
train_size = int(0.8 * num_samples)
test_size = num_samples - train_size
train_data = net_data[:train_size]
test_data = net_data[train_size:]

train_dataset = utils.TensorDataset(torch.from_numpy(train_data))
test_dataset = utils.TensorDataset(torch.from_numpy(test_data))

batch_size = 256
train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

def to_var(x, requires_grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

class GraphCNN(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphCNN, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
    def set_mask(self, mask):

        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True 
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
    def forward(self, x):
        if self.mask_flag:
            weight = self.weight * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features, in_features, kernel_size=3),
            nn.InstanceNorm2d(in_features)
        )
    def forward(self, x):
        return x + self.block(x)

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        latent_dim = 68
        input_nc = 1 
        init_depth = 48  
        
        z_size = latent_dim
        n_residual_blocks = 5  
        s_depth = 0
        ec = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, init_depth, 7),
            nn.InstanceNorm2d(init_depth),
            nn.ReLU(inplace=True)
        ]

        # Downsampling layers
        ec += [
            nn.Conv2d(init_depth, init_depth * 2, 3, stride=2, padding=1),
            nn.InstanceNorm2d(init_depth * 2),
            nn.ReLU(inplace=True)
        ]
        
        ec += [
            nn.Conv2d(init_depth * 2, init_depth * 4 - s_depth, 3, stride=2, padding=1),
            nn.InstanceNorm2d(init_depth * 4 - s_depth),
            nn.ReLU(inplace=True)
        ]
        
        in_features = init_depth * 4 - s_depth

        # Residual blocks
        for _ in range(n_residual_blocks):
            ec += [ResidualBlock(in_features)]
        
        self.ec = nn.Sequential(*ec)
        self.max_pool = nn.MaxPool2d(kernel_size=8, stride=8)
        flat_features = in_features * 2 * 2

        self.fc_mean = nn.Linear(flat_features, z_size)      
        self.fc_log_var = nn.Linear(flat_features, z_size)  

        self.fc3 = nn.Linear(latent_dim, 68)
        self.fc32 = nn.Linear(latent_dim,68)
        self.fc33 = nn.Linear(latent_dim,68)
        self.fc34 = nn.Linear(latent_dim,68)
        self.fc35 = nn.Linear(latent_dim,68)
        self.fc4 = nn.Linear(68,68)
        self.fc5 = nn.Linear(68,68)
        self.fc6 = nn.Linear(68,68)
        self.fc7 = nn.Linear(68,68)
        self.fc8 = nn.Linear(68,68)
        self.dropout = nn.Dropout(p=0.5)  
        # self.fcintercept = nn.Linear(68*68, 68*68)

    def encode(self, x):
        x = x.view(-1, 1, 68, 68)  
        x = self.ec(x)
        x = self.max_pool(x)
        x = x.view(x.size(0), -1) 
        mean = self.fc_mean(x)
        log_var = self.fc_log_var(x)
        return mean, log_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        # 使用sigmoid激活函数输出[0,1]
        h31 = torch.sigmoid(self.fc3(z))
        h31 = torch.sigmoid(self.fc4(h31))
        h31 = self.dropout(h31)
        h31_out = torch.bmm(h31.unsqueeze(2), h31.unsqueeze(1))

        h32 = torch.sigmoid(self.fc32(z))
        h32 = torch.sigmoid(self.fc5(h32))
        h32 = self.dropout(h32)
        h32_out = torch.bmm(h32.unsqueeze(2), h32.unsqueeze(1))

        h33 = torch.sigmoid(self.fc33(z))
        h33 = torch.sigmoid(self.fc6(h33))
        h33 = self.dropout(h33)
        h33_out = torch.bmm(h33.unsqueeze(2), h33.unsqueeze(1))

        h34 = torch.sigmoid(self.fc34(z))
        h34 = torch.sigmoid(self.fc7(h34))
        h34 = self.dropout(h34)
        h34_out = torch.bmm(h34.unsqueeze(2), h34.unsqueeze(1))

        h35 = torch.sigmoid(self.fc35(z))
        h35 = torch.sigmoid(self.fc8(h35))
        h35 = self.dropout(h35)
        h35_out = torch.bmm(h35.unsqueeze(2), h35.unsqueeze(1))

        h30 = torch.sigmoid(h31_out + h32_out + h33_out + h34_out + h35_out)
        h30 = h30.view(-1, 68*68)
        # 如果您不需要进一步线性层映射，可省略fcintercept
        # h30 = self.fcintercept(h30)

        return h30.view(-1, 68*68), h31+h32+h33+h34

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon, x_latent = self.decode(z)
        return recon.view(-1, 68 * 68), mu, logvar, x_latent


def loss_function(recon_x, x, mu, logvar):
    # 使用MSE loss适应FC数据
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KLD

model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 1000
train_losses = []

def train_epoch(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, _ = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_train_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average training loss: {:.4f}'.format(epoch, avg_train_loss))
    return avg_train_loss

for epoch in range(num_epochs):
    avg_loss = train_epoch(epoch)
    train_losses.append(avg_loss)

# 绘制并保存训练损失曲线
plt.figure()
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig('/home/yifzhang/Project/MDN_VAE/plot/training_loss_curve.png')
plt.close()

model.eval()
test_loader_for_metrics = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

generated_matrices = []
real_matrices = []

with torch.no_grad():
    for (batch_data,) in test_loader_for_metrics:
        batch_data = batch_data.to(device)
        recon_batch, _, _, _ = model(batch_data)
        recon_batch = recon_batch.cpu().numpy()

        for i in range(recon_batch.shape[0]):
            generated_matrix = recon_batch[i].reshape(68, 68)
            generated_matrices.append(generated_matrix)

        real_batch = batch_data.cpu().numpy()
        for i in range(real_batch.shape[0]):
            real_matrix = real_batch[i].reshape(68, 68)
            real_matrices.append(real_matrix)

generated_matrices = np.array(generated_matrices)
real_matrices = np.array(real_matrices)

print('Generated matrices shape:', generated_matrices.shape)
print('Real matrices shape:', real_matrices.shape)

real_matrices_fc = real_matrices
generated_matrices_fc = generated_matrices

def compute_mse(r, g):
    mse_values = ((r - g)**2).mean(axis=(1,2))
    return mse_values.mean()

mse_value = compute_mse(real_matrices_fc, generated_matrices_fc)
print('Average MSE:', mse_value)

def compute_psnr(r, g, max_value=1.0):
    mse_values = ((r - g)**2).mean(axis=(1,2))
    psnr_values = np.where(mse_values == 0, np.inf,
                           20 * np.log10(max_value) - 10 * np.log10(mse_values))
    return psnr_values[np.isfinite(psnr_values)].mean()

psnr_value = compute_psnr(real_matrices_fc, generated_matrices_fc, max_value=1.0)
print('Average PSNR:', psnr_value)

def compute_ssim_average(r, g):
    ssim_values = []
    for rm, gm in zip(r, g):
        val = ssim(rm, gm, data_range=1.0)
        ssim_values.append(val)
    return np.mean(ssim_values)

average_ssim = compute_ssim_average(real_matrices_fc, generated_matrices_fc)
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
real_features_spectral = [compute_spectral_features(m, k=k) for m in real_matrices_fc]
real_features_spectral = np.array(real_features_spectral)
generated_features_spectral = [compute_spectral_features(m, k=k) for m in generated_matrices_fc]
generated_features_spectral = np.array(generated_features_spectral)

mu_real_spectral = np.mean(real_features_spectral, axis=0)
sigma_real_spectral = np.cov(real_features_spectral, rowvar=False)
mu_gen_spectral = np.mean(generated_features_spectral, axis=0)
sigma_gen_spectral = np.cov(generated_features_spectral, rowvar=False)

fid_spectral = calculate_fid(mu_real_spectral, sigma_real_spectral, mu_gen_spectral, sigma_gen_spectral)
print('Spectral Based FID:', fid_spectral)

def compute_graph_features(matrix):
    max_val = np.max(np.abs(matrix))
    if max_val > 0:
        matrix_normalized = matrix / max_val
    else:
        matrix_normalized = np.zeros_like(matrix)
    # 使用 d = 1 - r
    distance_matrix = 1 - matrix_normalized

    G = nx.from_numpy_array(matrix_normalized)
    degrees = np.array([val for (node, val) in G.degree(weight='weight')])
    degree_mean = degrees.mean()
    degree_std = degrees.std()

    clustering_coeffs = np.array(list(nx.clustering(G, weight='weight').values()))
    clustering_mean = clustering_coeffs.mean()
    clustering_std = clustering_coeffs.std()

    edge_distances = {(i, j): distance_matrix[i, j] for i,j in G.edges()}
    nx.set_edge_attributes(G, edge_distances, 'distance')

    try:
        avg_shortest_path_length = nx.average_shortest_path_length(G, weight='distance')
    except nx.NetworkXError:
        lengths = []
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if len(subgraph) > 1:
                length = nx.average_shortest_path_length(subgraph, weight='distance')
                lengths.append(length)
        avg_shortest_path_length = np.mean(lengths) if lengths else 0

    features = np.array([
        degree_mean,
        degree_std,
        clustering_mean,
        clustering_std,
        avg_shortest_path_length
    ])
    return features

def calculate_fid_graph(features_real, features_generated):
    mu_real = np.mean(features_real, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)

    mu_gen = np.mean(features_generated, axis=0)
    sigma_gen = np.cov(features_generated, rowvar=False)

    diff = mu_real - mu_gen
    mean_diff = diff.dot(diff)

    covmean = sqrtm(sigma_real.dot(sigma_gen))
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = mean_diff + np.trace(sigma_real + sigma_gen - 2 * covmean)
    print(mean_diff)
    print(np.trace(sigma_real + sigma_gen - 2 * covmean))
    return fid

features_real_graph = [compute_graph_features(m) for m in real_matrices_fc]
features_generated_graph = [compute_graph_features(m) for m in generated_matrices_fc]

features_real_graph = np.array(features_real_graph)
features_generated_graph = np.array(features_generated_graph)

fid_graph = calculate_fid_graph(features_real_graph, features_generated_graph)

print("Graph Based FID:", fid_graph)

result_dir="/home/yifzhang/Project/MDN_VAE/result"
num_samples_to_save = 10
for i in range(num_samples_to_save):
    real_mat = real_matrices_fc[i]
    gen_mat = generated_matrices_fc[i]

    real_file_txt = os.path.join(result_dir, f"sample_{i}_real.txt")
    gen_file_txt = os.path.join(result_dir, f"sample_{i}_generated.txt")

    np.savetxt(real_file_txt, real_mat, fmt='%.6f')
    np.savetxt(gen_file_txt, gen_mat, fmt='%.6f')

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    im1 = axes[0].imshow(real_mat, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    axes[0].set_title(f"Real Sample {i}")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(gen_mat, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_title(f"Generated Sample {i}")
    plt.colorbar(im2, ax=axes[1])

    image_file = os.path.join(result_dir, f"sample_{i}_comparison.png")
    plt.tight_layout()
    plt.savefig(image_file)
    plt.close(fig)

print("Saved 10 samples of real and generated matrices and their comparisons in:", result_dir)
torch.cuda.empty_cache()
