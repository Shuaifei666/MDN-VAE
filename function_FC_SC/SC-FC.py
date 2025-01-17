from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import torch.utils.data as utils
from scipy.io import loadmat
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import os
from scipy.linalg import sqrtm

torch.manual_seed(11111)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 假设已存在成对 (FC, SC) 数据文件：
# ./data/desikan_fc_all.mat 包含 all_fc
# ./data/desikan_sc_all.mat 包含 all_sc
# 两者长度与索引对应
fc_mat = loadmat('./data/desikan_fc_all.mat')
all_fc_data = fc_mat['all_fc'][0]

sc_mat = loadmat('./data/desikan_sc_all.mat')
all_sc_data = sc_mat['all_sc'][0]

# 筛选出形状为(87,87)的 (FC, SC)
filtered_fc = []
filtered_sc = []
for fc_item, sc_item in zip(all_fc_data, all_sc_data):
    if fc_item.shape == (87,87) and sc_item.shape == (87,87):
        filtered_fc.append(fc_item)
        filtered_sc.append(sc_item)

fc_tensor = np.stack(filtered_fc, axis=-1) # (87,87,N)
sc_tensor = np.stack(filtered_sc, axis=-1) # (87,87,N)
N = fc_tensor.shape[2]

# FC: [-1,1] -> [0,1]
fc_data_list = []
for i in range(N):
    fc_mat = fc_tensor[:,:,i].astype(np.float32)
    fc_mat = (fc_mat+1.0)/2.0
    fc_data_list.append(fc_mat.flatten())
fc_data = np.array(fc_data_list, dtype=np.float32)

# SC: 假设非负，根据最大值归一化到[0,1]
max_sc = 0
for i in range(N):
    sc_mat = sc_tensor[:,:,i].astype(np.float32)
    max_sc = max(max_sc, sc_mat.max())
sc_data_list = []
for i in range(N):
    sc_mat = sc_tensor[:,:,i].astype(np.float32)/max_sc
    sc_data_list.append(sc_mat.flatten())
sc_data = np.array(sc_data_list, dtype=np.float32)

# 划分训练测试集
train_size = int(0.8*N)
test_size = N - train_size
train_fc = fc_data[:train_size]
train_sc = sc_data[:train_size]
test_fc = fc_data[train_size:]
test_sc = sc_data[train_size:]

train_dataset = utils.TensorDataset(torch.from_numpy(train_fc), torch.from_numpy(train_sc))
test_dataset = utils.TensorDataset(torch.from_numpy(test_fc), torch.from_numpy(test_sc))

batch_size = 256
train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 条件VAE结构
# Encoder输入: concat(FC, SC) -> mu, logvar
# Decoder输入: z和FC -> SC重构
in_dim = 68*68
hidden_dim = 1024
latent_dim = 64  # 可以根据需要调整

class cVAE(nn.Module):
    def __init__(self, in_dim=68*68, latent_dim=64):
        super(cVAE, self).__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        # encoder: 输入是FC和SC拼接，因此维度是 in_dim*2
        self.fc1 = nn.Linear(in_dim*2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # decoder: 输入是z和FC, concat后维度latent_dim+in_dim
        self.fc3 = nn.Linear(latent_dim+in_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, in_dim)

    def encode(self, fc, sc):
        x = torch.cat([fc, sc], dim=1)  # [batch, in_dim*2]
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, fc):
        x = torch.cat([z, fc], dim=1) # [batch, latent_dim+in_dim]
        h = F.relu(self.fc3(x))
        h = F.relu(self.fc4(h))
        out = torch.sigmoid(self.fc_out(h)) # SC in [0,1]
        return out

    def forward(self, fc, sc):
        mu, logvar = self.encode(fc, sc)
        z = self.reparameterize(mu, logvar)
        recon_sc = self.decode(z, fc)
        return recon_sc, mu, logvar

model = cVAE(in_dim=in_dim, latent_dim=latent_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

def loss_function(recon_sc, sc, mu, logvar):
    recon_loss = F.mse_loss(recon_sc, sc, reduction='sum')
    KLD = -0.5*torch.sum(1+logvar - mu.pow(2)-logvar.exp())
    return recon_loss + KLD, recon_loss, KLD

# 惩罚制度：让模型必须利用FC
# 在训练中随机将一半样本的FC置零，观察loss_no_fc
# 我们期望：loss_no_fc > loss_with_fc + margin
margin = 0.01

def train_epoch(epoch):
    model.train()
    total_loss = 0.0
    for fc_batch, sc_batch in train_loader:
        fc_batch = fc_batch.to(device)
        sc_batch = sc_batch.to(device)

        # 随机mask一半batch的FC
        batch_size = fc_batch.size(0)
        mask_idx = torch.randperm(batch_size)[:batch_size//2]
        fc_no = fc_batch.clone()
        fc_no[mask_idx] = 0.0  # 无FC信息

        # 前向计算有FC时的loss
        recon_sc, mu, logvar = model(fc_batch, sc_batch)
        loss_with_fc, rl_fc, kl_fc = loss_function(recon_sc, sc_batch, mu, logvar)

        # 前向计算无FC时的loss
        recon_no_fc, mu_no, logvar_no = model(fc_no, sc_batch)
        loss_no_fc, rl_no_fc, kl_no_fc = loss_function(recon_no_fc, sc_batch, mu_no, logvar_no)

        # 惩罚项：如果无FC时的loss不明显大于有FC时的loss + margin，则加罚
        reg = torch.relu(loss_with_fc - loss_no_fc + margin)
        loss = loss_with_fc + reg

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss/len(train_loader.dataset)
    print(f"Epoch {epoch}: Train Loss = {avg_loss:.6f}")
    return avg_loss

num_epochs = 200
train_losses = []
os.makedirs('/home/yifzhang/Project/MDN_VAE/plot', exist_ok=True)

for epoch in range(1,num_epochs+1):
    avg_loss = train_epoch(epoch)
    train_losses.append(avg_loss)

plt.figure()
plt.plot(range(1,num_epochs+1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()
plt.savefig('/home/yifzhang/Project/MDN_VAE/plot/train_loss_fc_sc_condition.png')
plt.close()

model.eval()
generated_matrices = []
real_matrices = []

with torch.no_grad():
    for fc_batch, sc_batch in test_loader:
        fc_batch = fc_batch.to(device)
        sc_batch = sc_batch.to(device)
        recon_sc, _, _ = model(fc_batch, sc_batch)
        gen_np = recon_sc.cpu().numpy()
        real_np = sc_batch.cpu().numpy()

        for i in range(gen_np.shape[0]):
            gen_mat = gen_np[i].reshape(68,68)
            real_mat = real_np[i].reshape(68,68)
            generated_matrices.append(gen_mat)
            real_matrices.append(real_mat)

generated_matrices = np.array(generated_matrices)
real_matrices = np.array(real_matrices)

print('Generated matrices shape:', generated_matrices.shape)
print('Real matrices shape:', real_matrices.shape)

def compute_mse(r, g):
    mse_values = ((r - g)**2).mean(axis=(1,2))
    return mse_values.mean()

mse_value = compute_mse(real_matrices, generated_matrices)
print('Average MSE:', mse_value)

def compute_psnr(r, g, max_value=1.0):
    mse_values = ((r - g)**2).mean(axis=(1,2))
    psnr_values = np.where(mse_values==0, np.inf, 20*np.log10(max_value)-10*np.log10(mse_values))
    return psnr_values[np.isfinite(psnr_values)].mean()

psnr_value = compute_psnr(real_matrices, generated_matrices, max_value=1.0)
print('Average PSNR:', psnr_value)

def compute_ssim_average(r, g):
    ssim_values = []
    for rm, gm in zip(r, g):
        val = ssim(rm, gm, data_range=1.0)
        ssim_values.append(val)
    return np.mean(ssim_values)

average_ssim = compute_ssim_average(real_matrices, generated_matrices)
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
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2*covmean)
    return fid

k = 30
real_features_spectral = [compute_spectral_features(m, k=k) for m in real_matrices]
real_features_spectral = np.array(real_features_spectral)
gen_features_spectral = [compute_spectral_features(m, k=k) for m in generated_matrices]
gen_features_spectral = np.array(gen_features_spectral)

mu_real = np.mean(real_features_spectral, axis=0)
sigma_real = np.cov(real_features_spectral, rowvar=False)
mu_gen = np.mean(gen_features_spectral, axis=0)
sigma_gen = np.cov(gen_features_spectral, rowvar=False)

fid_spectral = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
print('Spectral Based FID:', fid_spectral)

def compute_graph_features(matrix):
    max_val = matrix.max()
    if max_val>0:
        matrix_normalized = matrix/max_val
    else:
        matrix_normalized = np.zeros_like(matrix)
    epsilon = 1e-5
    distance_matrix = 1/(matrix_normalized+epsilon)
    G = nx.from_numpy_array(matrix_normalized)
    degrees = np.array([val for (_,val) in G.degree(weight='weight')])
    degree_mean = degrees.mean()
    degree_std = degrees.std()

    clustering_coeffs = np.array(list(nx.clustering(G, weight='weight').values()))
    clustering_mean = clustering_coeffs.mean()
    clustering_std = clustering_coeffs.std()

    try:
        avg_shortest_path_length = nx.average_shortest_path_length(G, weight='distance')
    except nx.NetworkXError:
        lengths=[]
        for component in nx.connected_components(G):
            subgraph = G.subgraph(component)
            if len(subgraph)>1:
                l = nx.average_shortest_path_length(subgraph, weight='distance')
                lengths.append(l)
        avg_shortest_path_length = np.mean(lengths) if lengths else 0

    features = np.array([
        degree_mean,
        degree_std,
        clustering_mean,
        clustering_std,
        avg_shortest_path_length
    ])
    return features

def calculate_fid_graph(features_real, features_gen):
    mu_real = np.mean(features_real, axis=0)
    sigma_real = np.cov(features_real, rowvar=False)
    mu_gen = np.mean(features_gen, axis=0)
    sigma_gen = np.cov(features_gen, rowvar=False)
    diff = mu_real - mu_gen
    mean_diff = diff.dot(diff)
    covmean = sqrtm(sigma_real.dot(sigma_gen))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = mean_diff + np.trace(sigma_real + sigma_gen -2*covmean)
    return fid

features_real_graph = [compute_graph_features(m) for m in real_matrices]
features_gen_graph = [compute_graph_features(m) for m in generated_matrices]
features_real_graph = np.array(features_real_graph)
features_gen_graph = np.array(features_gen_graph)

fid_graph = calculate_fid_graph(features_real_graph, features_gen_graph)
print("Graph Based FID:", fid_graph)

# 保存前10个样本对比
os.makedirs('/home/yifzhang/Project/MDN_VAE/result', exist_ok=True)
num_samples_to_save = min(10, len(generated_matrices))
for i in range(num_samples_to_save):
    real_mat = real_matrices[i]
    gen_mat = generated_matrices[i]

    real_file_txt = f"/home/yifzhang/Project/MDN_VAE/result/sample_{i}_real.txt"
    gen_file_txt = f"/home/yifzhang/Project/MDN_VAE/result/sample_{i}_generated.txt"
    np.savetxt(real_file_txt, real_mat, fmt='%.6f')
    np.savetxt(gen_file_txt, gen_mat, fmt='%.6f')

    fig, axes = plt.subplots(1, 2, figsize=(10,5))
    im1 = axes[0].imshow(real_mat, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    axes[0].set_title(f"Real SC {i}")
    plt.colorbar(im1, ax=axes[0])

    im2 = axes[1].imshow(gen_mat, cmap='hot', interpolation='nearest', vmin=0, vmax=1)
    axes[1].set_title(f"Generated SC {i}")
    plt.colorbar(im2, ax=axes[1])
    plt.tight_layout()
    image_file = f"/home/yifzhang/Project/MDN_VAE/result/sample_{i}_comparison.png"
    plt.savefig(image_file)
    plt.close(fig)

print("FC->SC generation completed and results saved.")
