from torchvision import datasets, transforms
from torch import utils
import matplotlib.pyplot as plt
import numpy as np
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))])

dataset_train = datasets.MNIST(
    '~/mnist',
    train=True,
    download=True,
    transform=transform)
dataset_valid = datasets.MNIST(
    '~/mnist',
    train=False,
    download=True,
    transform=transform)

dataloader_train = utils.data.DataLoader(dataset_train,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=4)
dataloader_valid = utils.data.DataLoader(dataset_valid,
                                          batch_size=1000,
                                          shuffle=True,
                                          num_workers=4)
import torch
import torch.nn as nn
import torch.nn.functional as F

device = 'cuda'


class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE, self).__init__()
        self.dense_enc1 = nn.Linear(28 * 28, 200)
        self.dense_enc2 = nn.Linear(200, 200)
        self.dense_encmean = nn.Linear(200, z_dim)
        self.dense_encvar = nn.Linear(200, z_dim)
        self.dense_dec1 = nn.Linear(z_dim, 200)
        self.dense_dec2 = nn.Linear(200, 200)
        self.dense_dec3 = nn.Linear(200, 28 * 28)

    def _encoder(self, x):
        x = F.relu(self.dense_enc1(x))
        x = F.relu(self.dense_enc2(x))
        mean = self.dense_encmean(x)
        var = F.softplus(self.dense_encvar(x))
        return mean, var

    def _sample_z(self, mean, var):
        epsilon = torch.randn(mean.shape).to(device)
        return mean + torch.sqrt(var) * epsilon

    def _decoder(self, z):
        x = F.relu(self.dense_dec1(z))
        x = F.relu(self.dense_dec2(x))
        x = F.sigmoid(self.dense_dec3(x))
        return x

    def forward(self, x):
        mean, var = self._encoder(x)
        z = self._sample_z(mean, var)
        x = self._decoder(z)
        return x, z

    def loss(self, x):
        mean, var = self._encoder(x)
        KL = -0.5 * torch.mean(torch.sum(1 + torch.log(var) - mean ** 2 - var))
        z = self._sample_z(mean, var)
        y = self._decoder(z)
        reconstruction = torch.mean(torch.sum(x * torch.log(y) + (1 - x) * torch.log(1 - y)))
        lower_bound = [-KL, reconstruction]
        return -sum(lower_bound)
import numpy as np
from torch import optim
import random
model = VAE(2).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
model.train()
for i in range(20):
  losses = []
  for x, t in dataloader_train:
      x = x.to(dtype=torch.float32, device=device)
      model.zero_grad()
      y = model(x)
      loss = model.loss(x)
      loss.backward()
      optimizer.step()
      losses.append(loss.cpu().detach().numpy())
  print("EPOCH: {} loss: {}".format(i, np.average(losses)))

fig = plt.figure(figsize=(20, 20))
model.eval()
zs = []

for x, t in dataloader_valid:
    x = x.to(dtype=torch.float32, device=device)
    y, z = model(x)
    dic=np.linspace(torch.min(z).cpu().detach().numpy(), torch.max(z).cpu().detach().numpy(), 20)
    counter=0
    for i in range(20):
        for j in range(20):
            counter+=1
            ax = fig.add_subplot(20, 20, counter, xticks=[], yticks=[])
            im= torch.tensor(model._decoder(torch.tensor([dic[i],dic[j]]).to(dtype=torch.float32, device=device))).cpu().detach().numpy()
            ax.imshow(im.reshape((28, 28, 1)), 'gray')
    break
plt.savefig('result.png')