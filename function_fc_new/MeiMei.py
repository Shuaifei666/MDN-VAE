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
import networkx as nx
from skimage.metrics import structural_similarity as ssim

torch.manual_seed(11111)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def fisher_z_transform(r, eps=1e-7):
    r = np.clip(r, -1+eps, 1-eps)
    return 0.5 * np.log((1+r)/(1-r))

def fisher_z_inverse(z):
    return (np.exp(2*z)-1)/(np.exp(2*z)+1)


dat_mat = loadmat('./data/desikan_fc_all.mat')
all_fc_data = dat_mat['all_fc'] 
fc_list = all_fc_data[0]
n_size = 32
filtered_fc = [mat for mat in fc_list if mat.shape == (87,87)]

tensor = np.stack(filtered_fc, axis=-1)
print("Tensor shape:", tensor.shape)  

A_mat = np.mean(tensor[18:86, 18:86, :], axis=2)


net_data = []
for i in range(tensor.shape[2]):
    ith = np.float32(tensor[:,:,i] + tensor[:,:,i].T)
    np.fill_diagonal(ith, np.mean(ith, 0))
    ith = ith[18:86, 18:86]
    ith_z = fisher_z_transform(ith)
    ith_z = ith_z.flatten()
    net_data.append(ith_z)
net_data = np.array(net_data, dtype=np.float32)
print("net_data shape after z-transform:", net_data.shape)


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
    def __init__(self, in_dim, out_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.relu = nn.ReLU(inplace=True)
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = None
    def forward(self, x):
        identity = x
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        if self.shortcut is not None:
            identity = self.shortcut(identity)
        out += identity
        out = self.relu(out)
        return out

class BranchMLP(nn.Module):
    def __init__(self, in_dim=68*68, out_dim=68, hidden_dims=[512, 512]):
        super(BranchMLP, self).__init__()
        self.init_fc = nn.Linear(in_dim, hidden_dims[0])
        blocks = []
        in_d = hidden_dims[0]
        for hdim in hidden_dims[1:]:
            blocks.append(ResidualBlock(in_d, hdim))
            in_d = hdim
        self.resblocks = nn.Sequential(*blocks)
        self.final_fc = nn.Linear(in_d, out_dim)

    def forward(self, x):
        h = self.init_fc(x)
        h = F.relu(h)
        h = self.resblocks(h)
        out = self.final_fc(h) 
        return out

class BranchGCN(nn.Module):
    def __init__(self, in_dim=68*68, n_nodes=68, num_gcn=3):
        super(BranchGCN, self).__init__()
        self.init_fc = nn.Linear(in_dim, n_nodes)
        self.gcn_layers = nn.ModuleList([GraphCNN(n_nodes, n_nodes) for _ in range(num_gcn)])
    def forward(self, x):
        h = self.init_fc(x)  
        h = F.relu(h)
        for gcn in self.gcn_layers:
            out = gcn(h)
            out = out + h
            out = F.relu(out)
            h = out
        return h

class MultiBranchDecoder(nn.Module):
 
    def __init__(self):
        super(MultiBranchDecoder, self).__init__()
        self.branch_mlp1 = BranchMLP(in_dim=68*68, out_dim=68, hidden_dims=[512,512])
        self.branch_gcn  = BranchGCN(in_dim=68*68, n_nodes=68, num_gcn=3)
        self.branch_mlp2 = BranchMLP(in_dim=68*68, out_dim=68, hidden_dims=[256,256])
        self.final_fc = nn.Linear(68*68, 68*68)

    def forward(self, x):
        out1 = self.branch_mlp1(x)  
        out2 = self.branch_gcn(x)  
        out3 = self.branch_mlp2(x)  

        merged = out1 + out2 + out3  
        merged_expanded = merged.unsqueeze(2)
        outer_68x68 = torch.bmm(merged_expanded, merged.unsqueeze(1))  

        flatten_out = outer_68x68.view(-1, 68*68)

        final_out = self.final_fc(flatten_out) 
        return final_out

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        latent_dim = 256
        self.fc11 = nn.Linear(68*68, 1024)
        self.fc12 = nn.Linear(68*68, 1024)
        self.fc111 = nn.Linear(1024,128)
        self.fc222 = nn.Linear(1024,128)
        self.fc21 = nn.Linear(128, latent_dim)
        self.fc22 = nn.Linear(128, latent_dim)

        self.fc_init_decode = nn.Linear(latent_dim, 68*68)  
        self.multi_branch_decoder = MultiBranchDecoder()    

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
        init_fc = self.fc_init_decode(z)  
        out_fc = self.multi_branch_decoder(init_fc)
        return out_fc  

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z) 
        return recon, mu, logvar


def loss_function(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KLD


model = VAE().to(device)



optimizer = optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 50
train_losses = []

def train_epoch(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
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

plt.figure()
plt.plot(range(num_epochs), train_losses, label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig('FC_MeiMei_training_loss_curve.png')
plt.close()

model.eval()
test_loader_for_metrics = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

generated_matrices = []
real_matrices = []

with torch.no_grad():
    for (batch_data,) in test_loader_for_metrics:
        batch_data = batch_data.to(device)
        recon_batch, mu, logvar = model(batch_data)
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

def compute_mse(r, g):
    mse_values = ((r - g)**2).mean(axis=(1,2))
    return mse_values.mean()

mse_value = compute_mse(real_matrices, generated_matrices)
print('Average MSE:', mse_value)

print("Done.")
