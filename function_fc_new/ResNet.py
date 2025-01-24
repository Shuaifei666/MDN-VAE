
from __future__ import print_function
import argparse
import torch
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
import torch.utils.data as utils
import matplotlib.pyplot as plt
from scipy.io import loadmat
from torch.autograd import Variable

torch.manual_seed(11111)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def fisher_z_transform(r, eps=1e-7):
    r = np.clip(r, -1+eps, 1-eps)  
    return 0.5 * np.log((1+r)/(1-r))

def fisher_z_inverse(z):
    return (np.exp(2*z)-1)/(np.exp(2*z)+1)


def load_data(matfile='./data/desikan_fc_all.mat', batch_size=256):
    dat_mat = loadmat(matfile)
    all_fc_data = dat_mat['all_fc']
    fc_list = all_fc_data[0]

    filtered_fc = [mat for mat in fc_list if mat.shape == (87,87)]
    tensor = np.stack(filtered_fc, axis=-1)  
    print("Tensor shape (raw):", tensor.shape)

    net_data = []
    for i in range(tensor.shape[2]):
        ith = np.float32(tensor[:,:,i] + tensor[:,:,i].T)
        np.fill_diagonal(ith, np.mean(ith, 0))
        ith = ith[18:86, 18:86]          
        ith_z = fisher_z_transform(ith)
        ith_z = ith_z.flatten()           
        net_data.append(ith_z)

    net_data = np.array(net_data, dtype=np.float32)
    print("Net_data shape after z-transform & flatten:", net_data.shape)


    num_samples = len(net_data)
    train_size = int(0.8 * num_samples)
    test_size = num_samples - train_size
    train_data = net_data[:train_size]
    test_data = net_data[train_size:]

    train_dataset = utils.TensorDataset(torch.from_numpy(train_data))
    test_dataset  = utils.TensorDataset(torch.from_numpy(test_data))

    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = utils.DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, net_data.shape[1]


class GraphCNN(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphCNN, self).__init__(in_features, out_features, bias)
        self.mask_flag = False

    def set_mask(self, mask, device=None):
        if device is not None:
            mask = mask.to(device)
        self.mask = mask
        with torch.no_grad():
            self.weight *= self.mask
        self.mask_flag = True

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

    def __init__(self, in_dim=68*68, out_dim=68,
                 decode_dimen_number=10,   
                 hidden_dim=256):
        super(BranchMLP, self).__init__()

        self.init_fc = nn.Linear(in_dim, hidden_dim)
        blocks = []
        for i in range(decode_dimen_number):
            blocks.append(ResidualBlock(hidden_dim, hidden_dim))
        self.resblocks = nn.Sequential(*blocks)
        self.final_fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        h = self.init_fc(x)
        h = F.relu(h)
        h = self.resblocks(h)
        out = self.final_fc(h)
        return out



class BranchGCN(nn.Module):

    def __init__(self, in_dim=68*68, n_nodes=68, num_gcn):
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
    def __init__(self,
                 decode_dimen_number_mlp1,
                 hidden_dim_mlp1,
                 decode_dimen_number_mlp2,
                 hidden_dim_mlp2,
                 decode_dimen_number_gcn):
        super(MultiBranchDecoder, self).__init__()

        self.branch_mlp1 = BranchMLP(
            in_dim=68*68, out_dim=68,
            decode_dimen_number=decode_dimen_number_mlp1,
            hidden_dim=hidden_dim_mlp1
        )

        self.branch_gcn = BranchGCN(
            in_dim=68*68, n_nodes=68,
            num_gcn=decode_dimen_number_gcn
        )

        self.branch_mlp2 = BranchMLP(
            in_dim=68*68, out_dim=68,
            decode_dimen_number=decode_dimen_number_mlp2,
            hidden_dim=hidden_dim_mlp2
        )

        self.final_fc = nn.Linear(68*68, 68*68)

    def forward(self, x):
        out1 = self.branch_mlp1(x)  # => [b,68]
        out2 = self.branch_gcn(x)   # => [b,68]
        out3 = self.branch_mlp2(x)  # => [b,68]

        merged = out1 + out2 + out3
        merged_expanded = merged.unsqueeze(2) # => [b,68,1]
        outer_68x68 = torch.bmm(merged_expanded, merged.unsqueeze(1)) # => [b,68,68]

        flatten_out = outer_68x68.view(-1, 68*68)
        final_out = self.final_fc(flatten_out)
        return final_out


class VAE(nn.Module):
    def __init__(self,
                 latent_dim,
                 decode_dimen_number_mlp1,
                 hidden_dim_mlp1,
                 decode_dimen_number_mlp2,
                 hidden_dim_mlp2,
                 decode_dimen_number_gcn):
        super(VAE, self).__init__()
        self.fc11 = nn.Linear(68*68, 1024)
        self.fc12 = nn.Linear(68*68, 1024)
        self.fc111 = nn.Linear(1024,128)
        self.fc222 = nn.Linear(1024,128)
        self.fc21 = nn.Linear(128, latent_dim)  # mu
        self.fc22 = nn.Linear(128, latent_dim)  # logvar

        self.fc_init_decode = nn.Linear(latent_dim, 68*68)
        self.multi_branch_decoder = MultiBranchDecoder(
            decode_dimen_number_mlp1=decode_dimen_number_mlp1,
            hidden_dim_mlp1=hidden_dim_mlp1,
            decode_dimen_number_mlp2=decode_dimen_number_mlp2,
            hidden_dim_mlp2=hidden_dim_mlp2,
            decode_dimen_number_gcn=decode_dimen_number_gcn
        )

    def encode(self, x):

        h11 = F.relu(self.fc11(x))
        h11 = F.relu(self.fc111(h11))
        h12 = F.relu(self.fc12(x))
        h12 = F.relu(self.fc222(h12))
        mu = self.fc21(h11)
        logvar = self.fc22(h12)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        init_fc = self.fc_init_decode(z)  # => [b,4624]
        out_fc = self.multi_branch_decoder(init_fc) # => [b,4624]
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

def main():
    from_datafile = './data/desikan_fc_all.mat'
    train_loader, test_loader, in_dim = load_data(matfile=from_datafile, batch_size=256)
    print("Data loaded. in_dim:", in_dim)  

    model = VAE(
        latent_dim=256,
        decode_dimen_number_mlp1=15,
        hidden_dim_mlp1=512,
        decode_dimen_number_mlp2=15,
        hidden_dim_mlp2=256,
        decode_dimen_number_gcn=10
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    num_epochs = 50
    train_losses = []
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data,) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss = loss_function(recon, data, mu, logvar)
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
        avg_loss = total_loss / len(train_loader.dataset)
        train_losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, AvgLoss={avg_loss:.4f}")

    plt.figure()
    plt.plot(range(num_epochs), train_losses, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.savefig('FC_VAE_training_loss_curve.png')
    plt.close()

    model.eval()
    test_generated = []
    test_real = []

    with torch.no_grad():
        for (batch_data,) in test_loader:
            batch_data = batch_data.to(device)
            recon, mu, logvar = model(batch_data)
            test_generated.append(recon.cpu().numpy())
            test_real.append(batch_data.cpu().numpy())

    test_generated = np.concatenate(test_generated, axis=0)
    test_real      = np.concatenate(test_real,      axis=0)

    print('test_generated shape:', test_generated.shape)
    print('test_real shape:', test_real.shape)

    mse_val = np.mean((test_generated - test_real)**2)
    print('Final MSE in z-domain:', mse_val)

    print("Done.")


if __name__ == "__main__":
    main()
