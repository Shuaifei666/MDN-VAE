from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import torch.utils.data as utils
import matplotlib.pyplot as plt
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
import networkx as nx
from scipy.linalg import sqrtm

torch.manual_seed(11111)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###################################
# 1. Data Loading
###################################
dat_mat = loadmat('./data/HCP_subcortical_CMData_desikan.mat')
tensor = dat_mat['loaded_tensor_sub']

A_mat = np.mean(np.squeeze(tensor[18:86, 18:86, 1, :]), axis=2)
A_mat = A_mat + A_mat.transpose()

loss_type = "poisson"
offset = 200 
n_size = 32

net_data = []
for i in range(tensor.shape[3]):
    ith = np.float32(tensor[:,:,0,i] + tensor[:,:,0,i].T)
    np.fill_diagonal(ith, np.mean(ith, 0))
    ith = ith[18:86, 18:86]
    ith = ith.flatten()
    ith = ith / offset
    net_data.append(ith)
net_data = np.array(net_data, dtype=np.float32)

num_samples = len(net_data)
train_size = int(0.8 * num_samples)
test_size = num_samples - train_size
train_data = net_data[:train_size]
test_data = net_data[train_size:]

batch_size = 256
train_dataset = utils.TensorDataset(torch.from_numpy(train_data))
test_dataset = utils.TensorDataset(torch.from_numpy(test_data))
train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

###################################
# 2. Build adjacency
###################################
def build_knn_adjacency(A_mat, k=32):
    n_nodes = A_mat.shape[0]
    adjacency = []
    for u in range(n_nodes):
        row = A_mat[u,:]
        indices = np.argsort(row)[::-1]
        topk = indices[:k]
        topk = topk[topk != u]
        topk = np.concatenate(([u], topk[:k-1]))
        adjacency.append(topk)
    return adjacency

adjacency_list = build_knn_adjacency(A_mat, k=n_size)
adjacency_list = [list(map(int, arr)) for arr in adjacency_list]
adjacency_list = np.array(adjacency_list, dtype=object)
print("Built adjacency with k=", n_size)

###################################
# 3. GCNLayer
###################################
class GCNLayer(nn.Module):
    def __init__(self, n_nodes=68, k=32):
        super().__init__()
        self.n_nodes = n_nodes
        self.k = k
        self.weight = nn.Parameter(torch.zeros(n_nodes, k))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(n_nodes))

    def forward(self, x_in, adjacency):
        batch_size = x_in.shape[0]
        x_out = torch.zeros_like(x_in)
        for u in range(self.n_nodes):
            nbh = adjacency[u]  
            w_u = self.weight[u]  
            x_nb = x_in[:, nbh]
            val = (x_nb * w_u).sum(dim=1) + self.bias[u]
            x_out[:, u] = val
        return torch.sigmoid(x_out)

###################################
# 4. One channel with multiple GCN layers
###################################
class ChannelGCN(nn.Module):
    def __init__(self, n_nodes=68, k=32, n_layers=3):
        super().__init__()
        self.n_layers = n_layers
        self.layers = nn.ModuleList([GCNLayer(n_nodes, k) for _ in range(n_layers)])
    
    def forward(self, x_in, adjacency):
        x = x_in
        for layer in self.layers:
            x = layer(x, adjacency)
        return x 

###################################
# 5. Multi-Channel GCN Decoder
###################################
class MultiChannelGCNDecoder(nn.Module):

    def __init__(self, R=5, n_nodes=68, k=32, n_layers=3, latent_dim=68):
        super().__init__()
        self.R = R
        self.n_nodes = n_nodes
        self.init_fc = nn.ModuleList([
            nn.Linear(latent_dim, n_nodes) for _ in range(R)
        ])
        self.channel_gcns = nn.ModuleList([
            ChannelGCN(n_nodes, k, n_layers) for _ in range(R)
        ])
        
    def forward(self, z, adjacency):
        batch_size = z.size(0)
        sc_sum = 0
        
        for r in range(self.R):
            x_r = self.init_fc[r](z)
            x_r = self.channel_gcns[r](x_r, adjacency)
            recon_r = torch.bmm(x_r.unsqueeze(2), x_r.unsqueeze(1))
            sc_sum = sc_sum + recon_r
        
        sc_sum = 0.5*(sc_sum + sc_sum.transpose(1,2))
        sc_flat = sc_sum.view(batch_size, -1)
        return sc_flat

###################################
# 6. VAE
###################################
class VAE(nn.Module):
    def __init__(self, R=5, n_nodes=68, k=32, n_layers=3, latent_dim=68):
        super(VAE, self).__init__()
        
        self.fc11 = nn.Linear(n_nodes*n_nodes, 1024)
        self.fc12 = nn.Linear(n_nodes*n_nodes, 1024)
        self.fc111 = nn.Linear(1024,128)
        self.fc222 = nn.Linear(1024,128)
        self.fc21 = nn.Linear(128, latent_dim)
        self.fc22 = nn.Linear(128, latent_dim)
        
        self.decoder = MultiChannelGCNDecoder(R=R, n_nodes=n_nodes, k=k,
                                              n_layers=n_layers, latent_dim=latent_dim)
    
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
    
    def decode(self, z, adjacency):
        return self.decoder(z, adjacency)
    
    def forward(self, x, adjacency):
        x = x.view(-1, 68*68)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, adjacency)
        return recon, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    BCE = F.poisson_nll_loss(recon_x, x, reduction='sum', log_input=True)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

###################################
# 7. Training / Testing
###################################
model = VAE(R=5, n_nodes=68, k=n_size, n_layers=3, latent_dim=68).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 500
train_losses = []

adjacency_list = build_knn_adjacency(A_mat, k=n_size)
adjacency_list = [list(map(int, arr)) for arr in adjacency_list]

def train_epoch(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data, adjacency_list)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_train_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch: {epoch}, Avg loss: {avg_train_loss:.4f}")
    return avg_train_loss

for epoch in range(num_epochs):
    avg_loss = train_epoch(epoch)
    train_losses.append(avg_loss)

plt.figure()
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('training_loss_curve.png')
plt.close()

# Test & evaluation
model.eval()
test_loader_for_metrics = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

generated_matrices = []
real_matrices = []

with torch.no_grad():
    for (batch_data,) in test_loader_for_metrics:
        batch_data = batch_data.to(device)
        recon_batch, mu, logvar = model(batch_data, adjacency_list)
        recon_batch = recon_batch.cpu().numpy()
        real_batch = batch_data.cpu().numpy()
        for i in range(recon_batch.shape[0]):
            gen_mat = recon_batch[i].reshape(68,68)
            generated_matrices.append(gen_mat)
            real_mat = real_batch[i].reshape(68,68)
            real_matrices.append(real_mat)

generated_matrices = np.array(generated_matrices)
real_matrices = np.array(real_matrices)
print('Generated matrices shape:', generated_matrices.shape)
print('Real matrices shape:', real_matrices.shape)


max_value = real_matrices.max()
real_matrices_norm = real_matrices / max_value
generated_matrices_norm = generated_matrices / max_value
generated_matrices_norm = np.maximum(generated_matrices_norm, 0)

def compute_mse(r, g):
    mse_values = ((r - g)**2).mean(axis=(1,2))
    return mse_values.mean()

mse_value = compute_mse(real_matrices_norm, generated_matrices_norm)
print('Average MSE:', mse_value)

def compute_psnr(r, g, max_value=1.0):
    mse_values = ((r - g)**2).mean(axis=(1,2))
    psnr_values = np.where(mse_values == 0, np.inf,
                           20 * np.log10(max_value) - 10 * np.log10(mse_values))
    return psnr_values[np.isfinite(psnr_values)].mean()

psnr_value = compute_psnr(real_matrices_norm, generated_matrices_norm, max_value=1.0)
print('Average PSNR:', psnr_value)

def compute_ssim_average(r, g):
    ssim_values = []
    for rm, gm in zip(r, g):
        ssim_value = ssim(rm, gm, data_range=1.0)
        ssim_values.append(ssim_value)
    return np.mean(ssim_values)

average_ssim = compute_ssim_average(real_matrices_norm, generated_matrices_norm)
print('Average SSIM:', average_ssim)

from scipy.linalg import sqrtm
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

k_spectral = 30
real_features = []
for matrix in real_matrices_norm:
    features = compute_spectral_features(matrix, k=k_spectral)
    real_features.append(features)
real_features = np.array(real_features)

generated_features = []
for matrix in generated_matrices_norm:
    features = compute_spectral_features(matrix, k=k_spectral)
    generated_features.append(features)
generated_features = np.array(generated_features)

mu_real = np.mean(real_features, axis=0)
sigma_real = np.cov(real_features, rowvar=False)
mu_gen = np.mean(generated_features, axis=0)
sigma_gen = np.cov(generated_features, rowvar=False)

fid_spectral = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
print('Spectral FID:', fid_spectral)

import networkx as nx
def compute_graph_features(matrix):
    max_val = np.max(np.abs(matrix))
    if max_val > 0:
        matrix_normalized = matrix / max_val
    else:
        matrix_normalized = np.zeros_like(matrix)
    epsilon = 1e-5
    distance_matrix = 1/(matrix_normalized+epsilon)
    G = nx.from_numpy_array(matrix_normalized)
    degrees = np.array([val for (node, val) in G.degree(weight='weight')])
    degree_mean = degrees.mean()
    degree_std = degrees.std()
    clustering_coeffs = np.array(list(nx.clustering(G, weight='weight').values()))
    clustering_mean = clustering_coeffs.mean()
    clustering_std = clustering_coeffs.std()
    try:
        avg_shortest_path_length = nx.average_shortest_path_length(G, weight='distance')
    except nx.NetworkXError:
        lengths = []
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if len(subgraph)>1:
                length = nx.average_shortest_path_length(subgraph, weight='distance')
                lengths.append(length)
        avg_shortest_path_length = np.mean(lengths) if lengths else 0
    features = np.array([degree_mean, degree_std, clustering_mean, clustering_std, avg_shortest_path_length])
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
    return fid

features_real_graph = [compute_graph_features(m) for m in real_matrices_norm]
features_generated_graph = [compute_graph_features(m) for m in generated_matrices_norm]
features_real_graph = np.array(features_real_graph)
features_generated_graph = np.array(features_generated_graph)

fid_graph = calculate_fid_graph(features_real_graph, features_generated_graph)
print("Graph FID:", fid_graph)
