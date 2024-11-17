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


device = torch.device("cuda")


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

dat_mat = loadmat('./data/HCP_subcortical_CMData_desikan.mat')
tensor = dat_mat['loaded_tensor_sub']
A_mat = np.mean(np.squeeze(tensor[:,:,1,:]), axis=2)
A_mat = A_mat + A_mat.transpose()
# A_mat[np.where(A_mat==0)] = 256


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        latent_dim = 87
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
        # self.fcintercept = nn.Linear(87*87, 87*87)
    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)
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
        h33 = F.sigmoid(self.fc5(h33))
        h33_out = torch.bmm(h33.unsqueeze(2), h33.unsqueeze(1))
        h34 = F.sigmoid(self.fc34(z))
        h34 = F.sigmoid(self.fc5(h34))
        h34_out = torch.bmm(h34.unsqueeze(2), h34.unsqueeze(1))
        h35 = F.sigmoid(self.fc35(z))
        h35 = F.sigmoid(self.fc5(h35))
        h35_out = torch.bmm(h35.unsqueeze(2), h35.unsqueeze(1))
        h30 = h31_out + h32_out + h33_out + h34_out + h35_out
        h30 = h30.view(-1, 87*87)
        # h30 = self.fcintercept(h30)
        return h30.view(-1, 87*87), h31+h32+h33
    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 87*87))
        z = self.reparameterize(mu, logvar)
        recon, x_latent = self.decode(z)
        return recon.view(-1, 87*87), mu, logvar, x_latent



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = VAE().to(device)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
batch_size = 256
learning_rate=0.001

net_data = []
for i in range(tensor.shape[3]):
    ith = np.float32(tensor[:,:,1,i] + np.transpose(tensor[:,:,1,i]))
    # ith[np.where(ith<=20)] = 0
    # ith[np.where(ith> 20)] = 1
    np.fill_diagonal(ith,np.mean(ith, 0))
    ith = ith.flatten()
    ith = np.log(ith+1)
    net_data.append(ith)

tensor_y = torch.stack([torch.Tensor(i) for i in net_data])
y = utils.TensorDataset(tensor_y) # create your datset
train_loader = utils.DataLoader(net_data, batch_size) 



def loss_function(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x , reduction='sum')
    BCE = F.poisson_nll_loss(recon_x , x, reduction='sum', log_input=True)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
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


for epoch in range(5000):
    train(epoch)
    with torch.no_grad():
        sample = torch.randn(16, 87).to(device)
        sample = model.decode(sample)[0].cpu()
        save_image(sample.view(16, 1, 87, 87),'results/sample_{}.png'.format(epoch))




torch.cuda.empty_cache()

latent_dim=87
num_elements = len(train_loader.dataset)
num_batches = len(train_loader)
batch_size = train_loader.batch_size
mu_out = torch.zeros(num_elements, latent_dim)
logvar_out = torch.zeros(num_elements,latent_dim)
recon_out = torch.zeros(num_elements,87*87)
x_latent_out = torch.zeros(num_elements, 87)
with torch.no_grad():
    for i, (data) in enumerate(train_loader):
        start = i*batch_size
        end = start + batch_size
        if i == num_batches - 1:
            end = num_elements
        data = data.to(device)
        recon_batch, mu, logvar, x_latent = model(data)
        mu_out[start:end] = mu
        logvar_out[start:end] = logvar
        x_latent_out[start:end] = x_latent
        recon_out[start:end] =recon_batch

with torch.no_grad():
    sample = torch.randn(10, 68).to(device)
    (sample,_) = model.decode(sample)
    sample = sample.cpu()
    for i in range(len(sample)):
        plt.imshow(sample[i].reshape(68,68))
        plt.savefig('results/{}.png'.format(i))

np.save('mu.npy', mu_out.detach().numpy())
import numpy, scipy.io
scipy.io.savemat('mu.mat', mdict={'mu': mu_out.detach().numpy()})

# torch.stack([torch.Tensor(i) for i in mu_stack])
