from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn, optim
from scipy.io import loadmat
import numpy as np
import torch.utils.data as utils
import matplotlib.pyplot as plt

# Set random seed for reproducibility
torch.manual_seed(11111)

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Load the dataset
dat_mat = loadmat('./data/HCP_samples.mat')
tensor = dat_mat['loaded_tensor_sub']

### Load and process the adjacency matrix
A_mat = np.mean(np.squeeze(tensor[18:86, 18:86, 1, :]), axis=2)
A_mat = A_mat + A_mat.transpose()

### Set the loss function type
loss_type = "MSE"

### Set the neighborhood size for GATE
n_size = 32

### Load network data and preprocess
net_data = []
for i in range(tensor.shape[3]):
    ith = np.float32(tensor[:, :, 0, i] + np.transpose(tensor[:, :, 0, i]))
    np.fill_diagonal(ith, np.mean(ith, axis=0))
    ith = ith[18:86, 18:86]
    ith = (ith - ith.min()) / (ith.max() - ith.min())
    ith = ith.flatten()
    net_data.append(ith)

batch_size = 5

# Convert network data to PyTorch tensors
tensor_y = torch.stack([torch.Tensor(i) for i in net_data])
y = utils.TensorDataset(tensor_y) 

# Create DataLoader
train_loader = utils.DataLoader(tensor_y, batch_size=batch_size, shuffle=True)

# Define the Residual Block used in the encoder
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

# Define the VAE model with the corrected encoder structure
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        latent_dim = 16  # Latent space dimension
        self.latent_dim = latent_dim  

        ### Parameters for the encoder structure
        input_nc = 1  # Number of input channels
        init_depth = 16  # Initial depth
        s_depth = 0  # Adjust this as needed
        n_residual_blocks = 2  # Number of residual blocks
        domains = 1  # Number of domains for embedding (set to 1 if not used)

        #### Encoder
        ec = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, init_depth, 7),
            nn.InstanceNorm2d(init_depth),
            nn.ReLU(inplace=True)
        ]

        # Downsampling
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

        #### Encoder Sequential Model
        self.ec = nn.Sequential(*ec)

        # Calculate the size after downsampling
        self.z_size = 68  # Initial size
        self.z_size = (self.z_size + 6 * 2) - 6  # After ReflectionPad2d and first Conv2d
        self.z_size = self.compute_output_size(self.z_size, kernel_size=7, stride=1, padding=0)
        self.z_size = self.compute_output_size(self.z_size, kernel_size=3, stride=2, padding=1)  # First downsampling
        self.z_size = self.compute_output_size(self.z_size, kernel_size=3, stride=2, padding=1)  # Second downsampling

        # Embedding layer (if domains > 1)
        self.s_depth = s_depth
        self.embed = nn.Embedding(domains, self.z_size * self.z_size * s_depth)

        # Additional convolutional layers
        self.z1 = nn.Conv2d(in_features, int((in_features + s_depth) / 2), 5, stride=4)
        self.tconv1 = nn.ConvTranspose2d(int((in_features + s_depth) / 2), int((in_features + s_depth) / 2),
                                         kernel_size=5, stride=4, padding=1, output_padding=2)
        self.tconv2 = nn.ConvTranspose2d(int((in_features + s_depth) / 2), in_features, kernel_size=1, stride=1)

        # Layers for latent variables
        self.encoder_fc_mu = nn.Linear(in_features * self.z_size * self.z_size, latent_dim)
        self.encoder_fc_logvar = nn.Linear(in_features * self.z_size * self.z_size, latent_dim)

        ### Decoder 
        self.decoder_fc1 = nn.Linear(latent_dim, 1024)
        self.decoder_fc2 = nn.Linear(1024, 64 * 9 * 9)
        self.decoder_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def compute_output_size(self, size, kernel_size, stride=1, padding=0):
        return (size - kernel_size + 2 * padding) // stride + 1

    def encode(self, x):
        """
        Encoder: Encodes the input data into latent variables (mu and logvar)
        """
        x = x.view(-1, 1, 68, 68)
        h = self.ec(x)
        # Flatten the output
        batch_size, channels, height, width = h.size()
        h = h.view(batch_size, -1)
        if not hasattr(self, 'encoder_fc_mu'):
            self.encoder_fc_mu = nn.Linear(channels * height * width, self.latent_dim).to(x.device)
            self.encoder_fc_logvar = nn.Linear(channels * height * width, self.latent_dim).to(x.device)
        mu = self.encoder_fc_mu(h)
        logvar = self.encoder_fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample z from N(mu, var)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """
        Decoder: Decodes the latent variable z back to the original data space
        """
        h = F.relu(self.decoder_fc1(z))        # (batch_size, 1024)
        h = F.relu(self.decoder_fc2(h))        # (batch_size, 64*9*9)
        h = h.view(-1, 64, 9, 9)               # (batch_size, 64, 9, 9)
        h = F.relu(self.decoder_conv1(h))      # (batch_size, 32, 17, 17)
        h = F.relu(self.decoder_conv2(h))      # (batch_size, 16, 33, 33)
        h = self.decoder_conv3(h)              # (batch_size, 1, 65, 65)
        h = torch.sigmoid(h)                   # Limit the output to [0, 1]
        h = F.interpolate(h, size=(68, 68), mode='bilinear', align_corners=False)  
        h = h.view(-1, 68 * 68)                # Flatten the output
        return h, z

    def forward(self, x):
        """
        Forward pass through the VAE
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x, z = self.decode(z)
        return recon_x, mu, logvar, z

# Define the loss function
def loss_function(recon_x, x, mu, logvar):
    """
    Compute the VAE loss function as the sum of reconstruction loss and KL divergence
    """
    # Use Mean Squared Error loss
    MSE = F.mse_loss(recon_x, x.view(-1, 68 * 68), reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # beta = 1.0
    return MSE + KLD

# Define the training function
def train(epoch):
    """
    Train the VAE model for one epoch
    """
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
    average_loss = train_loss / len(train_loader.dataset)
    print(f'====> Epoch: {epoch} Average loss: {average_loss:.4f}')

# Define the testing function
def test(epoch):
    """
    Evaluate the VAE model on the test set
    """
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(train_loader): 
            data = data.to(device)
            recon_batch, mu, logvar, _ = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
    test_loss /= len(train_loader.dataset)
    print(f'====> Test set loss: {test_loss:.4f}')

# Initialize the VAE model
model = VAE().to(device)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
num_epochs = 100
for epoch in range(1, num_epochs + 1):
    train(epoch)
    test(epoch)

# Generate and save samples after training
with torch.no_grad():
    sample = torch.randn(10, model.latent_dim).to(device)
    recon_samples, _ = model.decode(sample)
    recon_samples = recon_samples.cpu()
    sample = sample.cpu()

    sample_np = sample.numpy()
    recon_samples_np = recon_samples.numpy()

    for i in range(len(recon_samples)):
        latent_vector = sample_np[i]
        np.savetxt(f'results/latent_vector_{i}.csv', latent_vector, delimiter=',')
        reconstructed_sample = recon_samples_np[i]
        np.savetxt(f'results/reconstructed_sample_{i}.csv', reconstructed_sample, delimiter=',')
        plt.imshow(reconstructed_sample.reshape(68, 68))
        plt.axis('off')
        plt.savefig(f'results/sample_{i}.png')
        plt.close()
# Clear the CUDA cache
torch.cuda.empty_cache()
