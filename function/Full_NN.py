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
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def to_var(x, requires_grad=False, volatile=False):
    """
    Variable type that automatically chooses cpu or cuda
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
        self.weight.data = self.weight.data * self.mask.data
        self.mask_flag = True 
    def get_mask(self):
        print(self.mask_flag)
        return self.mask
    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight * self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)


dat_mat = loadmat('./data/HCP_subcortical_CMData_desikan.mat')
tensor = dat_mat['loaded_tensor_sub']
A_mat = np.mean(np.squeeze(tensor[:,:,1,:]), axis=2)
A_mat = A_mat + A_mat.transpose()

latent_dim = 87

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(87*87, 256)
        self.fc21 = nn.Linear(256, latent_dim)
        self.fc22 = nn.Linear(256, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 87)
        self.fc32 = nn.Linear(latent_dim,87)
        self.fc33 = nn.Linear(latent_dim,87)
        self.fc34 = nn.Linear(latent_dim,87)
        self.fc35 = nn.Linear(latent_dim,87)
        self.fc4 = nn.Linear(87,87)
        self.fc5 = nn.Linear(87,87)
        self.fc6 = nn.Linear(87,87)
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self, z):
        h31= torch.sigmoid(self.fc3(z))
        h31= torch.sigmoid(self.fc4(h31))
        h31_out = torch.bmm(h31.unsqueeze(2), h31.unsqueeze(1))
        h32 = torch.sigmoid(self.fc32(z))
        h32 = torch.sigmoid(self.fc5(h32))
        h32_out = torch.bmm(h32.unsqueeze(2), h32.unsqueeze(1))
        h33 = torch.sigmoid(self.fc33(z))
        h33 = torch.sigmoid(self.fc5(h33))
        h33_out = torch.bmm(h33.unsqueeze(2), h33.unsqueeze(1))
        h34 = torch.sigmoid(self.fc34(z))
        h34 = torch.sigmoid(self.fc5(h34))
        h34_out = torch.bmm(h34.unsqueeze(2), h34.unsqueeze(1))
        h35 = torch.sigmoid(self.fc35(z))
        h35 = torch.sigmoid(self.fc5(h35))
        h35_out = torch.bmm(h35.unsqueeze(2), h35.unsqueeze(1))
        h30 = h31_out + h32_out + h33_out + h34_out + h35_out
        h30 = h30.view(-1, 87*87)
        return h30.view(-1, 87*87), h31+h32+h33
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 87*87))
        z = self.reparameterize(mu, logvar)
        recon, x_latent = self.decode(z)
        return recon.view(-1, 87*87), mu, logvar, x_latent


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
batch_size = 256


# Prepare data
net_data = []
for i in range(tensor.shape[3]):
    ith = np.float32(tensor[:,:,1,i] + np.transpose(tensor[:,:,1,i]))
    np.fill_diagonal(ith, np.mean(ith, 0))
    ith = ith.flatten()
    ith = np.log(ith+1)
    net_data.append(ith)

net_data = np.array(net_data, dtype=np.float32)

# Split into train(80%) and test(20%)
num_samples = len(net_data)
train_size = int(0.8 * num_samples)
test_size = num_samples - train_size
train_data = net_data[:train_size]
test_data = net_data[train_size:]

train_dataset = utils.TensorDataset(torch.from_numpy(train_data))
test_dataset = utils.TensorDataset(torch.from_numpy(test_data))

train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = utils.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def loss_function(recon_x, x, mu, logvar):
    BCE = F.poisson_nll_loss(recon_x, x, reduction='sum', log_input=True)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


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

def test_epoch(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for (data,) in test_loader:
            data = data.to(device)
            recon_batch, mu, logvar, _ = model(data)
            loss = loss_function(recon_batch, data, mu, logvar).item()
            test_loss += loss
    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss


num_epochs = 500
train_losses = []
test_losses = []
for epoch in range(num_epochs):
    avg_train_loss = train_epoch(epoch)
    train_losses.append(avg_train_loss)

plt.figure()
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig('/home/yifzhang/Project/MDN_VAE/plot/fullnn_training_loss_curve.png')
plt.close()

model.eval()
test_data_array = test_data.astype(np.float32)
test_loader_for_metrics = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

generated_matrices = []
real_matrices = []

with torch.no_grad():
    for (batch_data,) in test_loader_for_metrics:
        batch_data = batch_data.to(device)
        recon_batch, _, _, _ = model(batch_data)
        recon_batch = recon_batch.cpu().numpy()

        for i in range(recon_batch.shape[0]):
            generated_matrix = recon_batch[i].reshape(87, 87)
            generated_matrices.append(generated_matrix)

        real_batch = batch_data.cpu().numpy()
        for i in range(real_batch.shape[0]):
            real_matrix = real_batch[i].reshape(87, 87)
            real_matrices.append(real_matrix)

generated_matrices = np.array(generated_matrices)
real_matrices = np.array(real_matrices)

print('Generated matrices shape:', generated_matrices.shape)
print('Real matrices shape:', real_matrices.shape)

max_value = real_matrices.max()
real_matrices_norm = real_matrices / max_value
generated_matrices_norm = generated_matrices / max_value
generated_matrices_norm = np.maximum(generated_matrices_norm, 0)

def compute_mse(real_matrices, generated_matrices):
    mse_values = ((real_matrices - generated_matrices) ** 2).mean(axis=(1, 2))
    average_mse = mse_values.mean()
    return average_mse

mse_value = compute_mse(real_matrices_norm, generated_matrices_norm)
print('Average MSE:', mse_value)

def compute_psnr(real_matrices, generated_matrices, max_value=1.0):
    mse_values = ((real_matrices - generated_matrices) ** 2).mean(axis=(1, 2))
    psnr_values = np.where(mse_values == 0, np.inf,
                           20 * np.log10(max_value) - 10 * np.log10(mse_values))
    average_psnr = psnr_values[np.isfinite(psnr_values)].mean()
    return average_psnr

psnr_value = compute_psnr(real_matrices_norm, generated_matrices_norm, max_value=1.0)
print('Average PSNR:', psnr_value)

def compute_ssim_average(real_matrices, generated_matrices):
    ssim_values = []
    for real_mat, gen_mat in zip(real_matrices, generated_matrices):
        ssim_value = ssim(real_mat, gen_mat, data_range=1.0)
        ssim_values.append(ssim_value)
    average_ssim = np.mean(ssim_values)
    return average_ssim

average_ssim = compute_ssim_average(real_matrices_norm, generated_matrices_norm)
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

k = 10

real_features = []
for matrix in real_matrices_norm:
    features = compute_spectral_features(matrix, k=k)
    real_features.append(features)
real_features = np.array(real_features)

generated_features = []
for matrix in generated_matrices_norm:
    features = compute_spectral_features(matrix, k=k)
    generated_features.append(features)
generated_features = np.array(generated_features)

mu_real = np.mean(real_features, axis=0)
sigma_real = np.cov(real_features, rowvar=False)

mu_gen = np.mean(generated_features, axis=0)
sigma_gen = np.cov(generated_features, rowvar=False)

fid_value = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
print('FID value:', fid_value)

torch.cuda.empty_cache()
