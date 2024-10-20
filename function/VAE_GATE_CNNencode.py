from __future__ import print_function
import torch
import torch.nn.functional as F
from torch import nn, optim
from scipy.io import loadmat
import numpy as np
import torch.utils.data as utils
import matplotlib.pyplot as plt

# Remove the random seed setting to allow generating diverse samples
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
    # Normalize data to the [0, 1] range
    ith = (ith - ith.min()) / (ith.max() - ith.min())
    ith = ith.flatten()
    net_data.append(ith)

batch_size = 5

# Convert network data to PyTorch tensors
tensor_y = torch.stack([torch.Tensor(i) for i in net_data])
y = utils.TensorDataset(tensor_y)  # Create the dataset

# Create DataLoader
train_loader = utils.DataLoader(tensor_y, batch_size=batch_size, shuffle=True)

# Define the VAE model
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        latent_dim = 16  # Increase the latent space dimension

        ### Encoder
        self.encoder_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)   # (1,68,68) -> (16,34,34)
        self.encoder_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)  # (16,34,34) -> (32,17,17)
        self.encoder_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)  # (32,17,17) -> (64,9,9)
        self.encoder_fc1 = nn.Linear(64 * 9 * 9, 1024)
        self.encoder_fc_mu = nn.Linear(1024, latent_dim)
        self.encoder_fc_logvar = nn.Linear(1024, latent_dim)

        ### Decoder (Fully Connected Network)
        self.decoder_fc1 = nn.Linear(latent_dim, 1024)
        self.decoder_fc2 = nn.Linear(1024, 64 * 9 * 9)
        self.decoder_conv1 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv2 = nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder_conv3 = nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def encode(self, x):
        """
        Encoder: Encodes the input data into latent variables (mu and logvar)
        """
        x = x.view(-1, 1, 68, 68)
        h = F.relu(self.encoder_conv1(x))  # (16,34,34)
        h = F.relu(self.encoder_conv2(h))  # (32,17,17)
        h = F.relu(self.encoder_conv3(h))  # (64,9,9)
        h = h.view(-1, 64 * 9 * 9)
        h = F.relu(self.encoder_fc1(h))    # (batch_size, 1024)
        mu = self.encoder_fc_mu(h)         # (batch_size, latent_dim)
        logvar = self.encoder_fc_logvar(h) # (batch_size, latent_dim)
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
        h = F.interpolate(h, size=(68, 68), mode='bilinear', align_corners=False)  # Resize to original dimensions
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
    # KL divergence
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Adjust the weight of KL divergence
    beta = 1.0
    return MSE + beta * KLD

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
        for i, data in enumerate(train_loader):  # Ideally, use a separate test_loader
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
num_epochs = 200
for epoch in range(1, num_epochs + 1):
    train(epoch)
    test(epoch)

# Generate and save samples after training
with torch.no_grad():
    sample = torch.randn(10, 16).to(device)
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
