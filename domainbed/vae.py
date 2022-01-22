from __future__ import print_function
import torch 
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from copy import deepcopy
import argparse
import numpy as np

class VAE(nn.Module):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.fc1 = nn.Linear(1568, 200) # The dimensions of VAE need to be changed for images with different shape
        self.fc21 = nn.Linear(200, latent_dim) 
        self.fc22 = nn.Linear(200, latent_dim)
        self.fc3 = nn.Linear(latent_dim, 200)
        self.fc4 = nn.Linear(200, 1568)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, 1568))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def klbo_loss(recon_x, x, mu, logvar):
    BCE = F.binary_cross_entropy(recon_x, x.view(-1, 1568), reduction='sum')

    KLD = -0.5*torch.sum(1+logvar-mu.pow(2)-logvar.exp())
    return BCE + KLD

def vae_train(vae, train_loader):
    device = "cuda"
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()
    for step in range(5000):
        train_loss = 0
        minibatch = [x.to(device) for x,y in next(train_loader)] # [n_envs, batch]
        minibatch = torch.stack(minibatch)
        data = minibatch.view(-1, 2, 28, 28)
        optimizer.zero_grad()
        recon_batch, mu, logvar = vae(data)
        loss = klbo_loss(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()


def soft_arg_max(A, mask=None, env=None, beta=10, dim=1, epsilon=1e-6):
    A_max,_ = torch.max(A, dim=dim, keepdim=True)
    
    A_exp = torch.exp((A-A_max)*beta)
    
    assert not torch.isnan(A).any(), 'A is nan'
    assert not torch.isnan(A_exp).any(), 'A_exp is nan'
    
    A_softmax = A_exp/(torch.sum(A_exp, dim=dim, keepdim=True) + epsilon)
    if mask is not None:
        A_softmax = A*mask
    #print(A_softmax)
    indices = torch.arange(start=0, end=A.size()[dim]).float().cuda()
    result =  torch.matmul(A_softmax, indices)
    #print(result)
    assert not torch.isnan(result).any()
    return torch.matmul(A_softmax, indices)

def mean_nll(logits, y):
    return nn.functional.binary_cross_entropy_with_logits(logits, y)

def reluloss_coef(logits, y, coef=None):
    scale = torch.tensor(1.).cuda().requires_grad_()
    tmploss = nn.functional.cross_entropy(logits*scale, y.squeeze(), reduction='none')
    
    coef = coef.squeeze()
    tmploss = tmploss.squeeze()

    
    loss =  tmploss*coef
    '''
    print(coef)
    print(loss.shape)
    print(coef.shape)
    print(newloss.shape)
    #print('=====================')
    print(newloss)
    exit()
    '''
    loss = torch.mean(loss)

    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]

    penalty = torch.sum(grad**2)

    return penalty