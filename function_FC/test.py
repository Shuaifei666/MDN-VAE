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
import os

# 额外导入
from skimage.metrics import structural_similarity as ssim
import networkx as nx
from scipy.linalg import sqrtm

torch.manual_seed(11111)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------
# 1. 数据加载
dat_mat = loadmat('./data/desikan_fc_all.mat')
all_fc_data = dat_mat['all_fc'] 
fc_list = all_fc_data[0]
n_size = 32

# 只筛选 (87,87) 尺寸
filtered_fc = [mat for mat in fc_list if mat.shape == (87,87)]
tensor = np.stack(filtered_fc, axis=-1)   # shape: (87,87,N)
print("Tensor shape:", tensor.shape)

# 计算 A_mat, 用于后面 mask
A_mat = np.mean(tensor[18:86, 18:86, :], axis=2)

# 将 FC 从 [-1,1] 映射到 [0,1]
net_data = []
for i in range(tensor.shape[2]):
    ith = np.float32(tensor[:,:,i] + tensor[:,:,i].T)
    np.fill_diagonal(ith, np.mean(ith, 0))
    ith = ith[18:86, 18:86]
    ith = (ith + 1.0) / 2.0  # [-1,1]->[0,1]
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

# ---------------------------------------------------------------------
# 2. 定义 GraphCNN 与 VAE 模型

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

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        latent_dim = 68
        # 编码部分
        self.fc11 = nn.Linear(68*68, 1024)
        self.fc12 = nn.Linear(68*68, 1024)
        self.fc111 = nn.Linear(1024,128)
        self.fc222 = nn.Linear(1024,128)
        self.fc21 = nn.Linear(128, latent_dim)
        self.fc22 = nn.Linear(128, latent_dim)

        # 解码部分
        self.fc3 = nn.Linear(latent_dim, 68)
        self.fc32 = nn.Linear(latent_dim,68)
        self.fc33 = nn.Linear(latent_dim,68)
        self.fc34 = nn.Linear(latent_dim,68)
        self.fc35 = nn.Linear(latent_dim,68)

        self.fc4 = GraphCNN(68,68)
        self.fc5 = GraphCNN(68,68)
        self.fc6 = GraphCNN(68,68)
        self.fc7 = GraphCNN(68,68)
        self.fc8 = GraphCNN(68,68)
        self.fcintercept = GraphCNN(68*68, 68*68)

    def encode(self, x):
        # x shape: [batch, 68*68]
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
        # intermediate
        h31 = torch.tanh(self.fc3(z))
        h31 = torch.tanh(self.fc4(h31))
        # outer product
        h31_out = torch.bmm(h31.unsqueeze(2), h31.unsqueeze(1))

        h32 = torch.tanh(self.fc32(z))
        h32 = torch.tanh(self.fc5(h32))
        h32_out = torch.bmm(h32.unsqueeze(2), h32.unsqueeze(1))

        h33 = torch.tanh(self.fc33(z))
        h33 = torch.tanh(self.fc6(h33))
        h33_out = torch.bmm(h33.unsqueeze(2), h33.unsqueeze(1))

        h34 = torch.tanh(self.fc34(z))
        h34 = torch.tanh(self.fc7(h34))
        h34_out = torch.bmm(h34.unsqueeze(2), h34.unsqueeze(1))

        h35 = torch.tanh(self.fc35(z))
        h35 = torch.tanh(self.fc8(h35))
        h35_out = torch.bmm(h35.unsqueeze(2), h35.unsqueeze(1))

        h30 = torch.tanh(h31_out + h32_out + h33_out + h34_out + h35_out)
        h30 = h30.view(-1, 68*68)
        h30 = self.fcintercept(h30)  # remain in [-1,1]
        return h30.view(-1, 68*68), (h31+h32+h33+h34)

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1,68*68))
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

# ---------------------------------------------------------------------
# 3. 定义损失与模型初始化
def loss_function(recon_x, x, mu, logvar):
    # x 与 recon_x 都是 [batch, 68*68]
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + KLD

model = VAE().to(device)

# 构造mask
masks = []
mask_2NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_2NN)).float())
mask_4NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_4NN)).float())
mask_8NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_8NN)).float())
mask_16NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_16NN)).float())
mask_32NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_32NN)).float())
mask_intercept = np.identity(68*68)
masks.append(torch.from_numpy(np.float32(mask_intercept)).float())
model.set_mask(masks)

optimizer = optim.Adam(model.parameters(), lr=1e-3)

# ---------------------------------------------------------------------
# 4. 训练过程 (示例只跑几个epoch)
def train_epoch(epoch):
    model.train()
    total_loss=0
    for batch_idx, (data,) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon, mu, logvar, x_latent = model(data)
        loss = loss_function(recon, data, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    avg_loss = total_loss / len(train_loader.dataset)
    print(f"Epoch {epoch}: Train loss: {avg_loss:.6f}")
    return avg_loss

num_epochs=3  # 可根据需求增加
train_losses=[]
for epoch in range(1,num_epochs+1):
    avg_loss = train_epoch(epoch)
    train_losses.append(avg_loss)

# 绘制并保存训练损失
os.makedirs("/home/yifzhang/Project/MDN_VAE/plot", exist_ok=True)
plt.figure()
plt.plot(range(1,num_epochs+1), train_losses, label='Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()
plt.savefig('/home/yifzhang/Project/MDN_VAE/plot/fc_vae_training_loss_curve.png')
plt.close()

# ---------------------------------------------------------------------
# 5. Debug: 单次前向传播并逐层打印输出
def debug_forward_pass(input_data):
    """
    输入 input_data: shape [1, 68*68], [batch=1, feature=4624]
    打印每层输出 shape 及 (min, max) 值.
    """
    print("\n========== Debug Forward Pass ==========")
    # 先看encode:
    #  input_data -> fc11, fc111, fc12, fc222 -> mu, logvar
    #  这里我们手动拆分, 并打印 shape / range

    print("[Input]", input_data.shape, 
          "range:", (input_data.min().item(), input_data.max().item()))
    x_reshaped = input_data.view(-1,68*68)  # (1,4624)

    # fc11
    h11 = F.relu(model.fc11(x_reshaped))
    print("[fc11->relu]", h11.shape,
          "range:", (h11.min().item(), h11.max().item()))
    h11 = F.relu(model.fc111(h11))
    print("[fc111->relu]", h11.shape,
          "range:", (h11.min().item(), h11.max().item()))

    # fc12
    h12 = F.relu(model.fc12(x_reshaped))
    print("[fc12->relu]", h12.shape,
          "range:", (h12.min().item(), h12.max().item()))
    h12 = F.relu(model.fc222(h12))
    print("[fc222->relu]", h12.shape,
          "range:", (h12.min().item(), h12.max().item()))

    mu = model.fc21(h11)
    logvar = model.fc22(h12)
    print("[mu]", mu.shape, "range:", (mu.min().item(), mu.max().item()))
    print("[logvar]", logvar.shape, "range:", (logvar.min().item(), logvar.max().item()))

    # reparameterize
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    z = mu + eps*std
    print("[z]", z.shape, "range:", (z.min().item(), z.max().item()))

    # decode
    # fc3->fc4->tanh
    h31 = torch.tanh(model.fc3(z))
    h31 = torch.tanh(model.fc4(h31))
    print("[fc3->fc4->tanh]", h31.shape, "range:", (h31.min().item(), h31.max().item()))

    # outer product
    h31_out = torch.bmm(h31.unsqueeze(2), h31.unsqueeze(1))
    print("[h31_out outer product]", h31_out.shape, 
          "range:", (h31_out.min().item(), h31_out.max().item()))

    # 其余 h32, h33... 不再逐一展开, 您可以酌情添加
    # 最后 fcintercept
    h30 = torch.tanh(h31_out)
    h30 = h30.view(-1,68*68)
    h30 = model.fcintercept(h30)
    print("[fcintercept output]", h30.shape,
          "range:", (h30.min().item(), h30.max().item()))

# ---------------------------------------------------------------------
# 6. 从测试集中选一个样本进行 debug
test_iter = iter(test_loader)
example_batch, = next(test_iter)
example_data = example_batch[0].unsqueeze(0).to(device)  # 取1个样本
print("\nExample data shape:", example_data.shape)

debug_forward_pass(example_data)
