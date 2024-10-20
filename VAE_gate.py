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
import matplotlib.pyplot as plt
import time
import os
print("Starting VAE_gate.py")

# Set the random seed for reproducibility
torch.manual_seed(11111)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Load data set
dat_mat = loadmat('./data/HCP_samples.mat')
tensor = dat_mat['loaded_tensor_sub']

### Load the adjacency matrix
A_mat = np.mean(np.squeeze(tensor[18:86, 18:86, 1, :]), axis=2)
A_mat = A_mat + A_mat.transpose()

### Choose the proper type of loss function 
loss_type = "poisson"

### If the loss_type = "poisson", set the proper offset for the mean
offset = 100

### Set the neighborhood size for GATE
n_size = 32

### Prepare the dataset
net_data = []
for i in range(tensor.shape[3]):
    ith = np.float32(tensor[:, :, 0, i] + np.transpose(tensor[:, :, 0, i]))
    np.fill_diagonal(ith, np.mean(ith, 0))
    ith = ith[18:86, 18:86]
    ith = ith.flatten()
    ith = ith / offset
    net_data.append(ith)

batch_size = 5

# Create Dataset
class NetDataset(utils.Dataset):
    def __init__(self, data_list):
        self.data = [torch.Tensor(i) for i in data_list]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        return sample

train_dataset = NetDataset(net_data)
train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

def to_var(x, requires_grad=False):
    """
    Variable type that automatically chooses CPU or GPU
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)

# Define the ResidualBlock as in the Generator class
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

# GraphCNN class as provided
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

# VAE class with the encoder replaced
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        latent_dim = 8
        input_nc = 1  # Single-channel input
        output_nc = 1
        init_depth = 64
        s_depth = 16
        bc = True
        n_residual_blocks = 9
        self.z_size = 17 if bc else 7

        #### Encoder (from the Generator class)
        ec = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, init_depth, kernel_size=7),
            nn.InstanceNorm2d(init_depth),
            nn.ReLU(inplace=True),
            # Downsampling
            nn.Conv2d(init_depth, init_depth * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(init_depth * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(init_depth * 2, init_depth * 4 - s_depth, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(init_depth * 4 - s_depth),
            nn.ReLU(inplace=True)
        ]
        in_features = init_depth * 4 - s_depth

        # Residual blocks
        for _ in range(n_residual_blocks):
            ec += [ResidualBlock(in_features)]

        self.ec = nn.Sequential(*ec)

        # Compute mu and logvar
        self.fc_mu = nn.Linear(in_features, latent_dim)
        self.fc_logvar = nn.Linear(in_features, latent_dim)

        # Decoder layers (unchanged)
        self.fc3 = nn.Linear(latent_dim, 68)
        self.fc32 = nn.Linear(latent_dim, 68)
        self.fc33 = nn.Linear(latent_dim, 68)
        self.fc34 = nn.Linear(latent_dim, 68)
        self.fc35 = nn.Linear(latent_dim, 68)
        self.fc4 = GraphCNN(68, 68)
        self.fc5 = GraphCNN(68, 68)
        self.fc6 = GraphCNN(68, 68)
        self.fc7 = GraphCNN(68, 68)
        self.fc8 = GraphCNN(68, 68)
        self.fcintercept = GraphCNN(68 * 68, 68 * 68)

    def encode(self, x):
        x = x.view(-1, 1, 68, 68)  # Reshape input
        x = self.ec(x)
        x = F.adaptive_avg_pool2d(x, 1)  # Global average pooling
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h31 = torch.sigmoid(self.fc3(z))
        h31 = torch.sigmoid(self.fc4(h31))
        h31_out = torch.bmm(h31.unsqueeze(2), h31.unsqueeze(1))
        h32 = torch.sigmoid(self.fc32(z))
        h32 = torch.sigmoid(self.fc5(h32))
        h32_out = torch.bmm(h32.unsqueeze(2), h32.unsqueeze(1))
        h33 = torch.sigmoid(self.fc33(z))
        h33 = torch.sigmoid(self.fc6(h33))
        h33_out = torch.bmm(h33.unsqueeze(2), h33.unsqueeze(1))
        h34 = torch.sigmoid(self.fc34(z))
        h34 = torch.sigmoid(self.fc7(h34))
        h34_out = torch.bmm(h34.unsqueeze(2), h34.unsqueeze(1))
        h35 = torch.sigmoid(self.fc35(z))
        h35 = torch.sigmoid(self.fc8(h35))
        h35_out = torch.bmm(h35.unsqueeze(2), h35.unsqueeze(1))
        h30 = torch.sigmoid(h31_out + h32_out + h33_out + h34_out + h35_out)
        h30 = h30.view(-1, 68 * 68)
        h30 = self.fcintercept(h30)
        return h30, h31 + h32 + h33 + h34 + h35

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon, x_latent = self.decode(z)
        return recon, mu, logvar, x_latent

    def set_mask(self, masks):
        self.fc4.set_mask(masks[0])
        self.fc5.set_mask(masks[1])
        self.fc6.set_mask(masks[2])
        self.fc7.set_mask(masks[3])
        self.fc8.set_mask(masks[4])
        self.fcintercept.set_mask(masks[5])

# Loss function remains the same
def loss_function(recon_x, x, mu, logvar):
    if loss_type == "poisson":
        x = x + offset  # Adjust input for Poisson loss
        BCE = F.poisson_nll_loss(recon_x, x, reduction='sum', log_input=True)
    elif loss_type == "binary":
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    else:
        raise ValueError("Unsupported loss type")
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

# Training function
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar, _ = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    avg_loss = train_loss / len(train_loader.dataset)
    print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, avg_loss))

# Testing function
def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(train_loader):
            data = data.to(device)
            recon_batch, mu, logvar, _ = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(train_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))

# Initialize the model and set masks
model = VAE().to(device)

# Prepare masks as per your original code
masks = []
mask_2NN = (np.argsort(np.argsort(A_mat, axis=-1), axis=-1) < n_size + 1)
masks.append(torch.from_numpy(np.float32(mask_2NN)).float())
mask_4NN = (np.argsort(np.argsort(A_mat, axis=-1), axis=-1) < n_size + 1)
masks.append(torch.from_numpy(np.float32(mask_4NN)).float())
mask_8NN = (np.argsort(np.argsort(A_mat, axis=-1), axis=-1) < n_size + 1)
masks.append(torch.from_numpy(np.float32(mask_8NN)).float())
mask_16NN = (np.argsort(np.argsort(A_mat, axis=-1), axis=-1) < n_size + 1)
masks.append(torch.from_numpy(np.float32(mask_16NN)).float())
mask_16NN = (np.argsort(np.argsort(A_mat, axis=-1), axis=-1) < n_size + 1)
masks.append(torch.from_numpy(np.float32(mask_16NN)).float())
mask_intercept = np.identity(68 * 68)
masks.append(torch.from_numpy(np.float32(mask_intercept)).float())
model.set_mask(masks)

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(1, 101):
    train(epoch)
    test(epoch)

# Generate and save reconstructed images
with torch.no_grad():
    sample = torch.randn(10, 8).to(device)
    sample, _ = model.decode(sample)
    sample = sample.cpu()
    if not os.path.exists('results'):
        os.makedirs('results')
    for i in range(len(sample)):
        plt.imshow(sample[i].view(68, 68).numpy(), cmap='gray')
        plt.savefig('results/{}.png'.format(i))
        plt.close()

# Clear CUDA cache
torch.cuda.empty_cache()
