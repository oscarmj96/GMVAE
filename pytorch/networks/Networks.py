"""
---------------------------------------------------------------------
-- Author: Jhosimar George Arias Figueroa
---------------------------------------------------------------------

Gaussian Mixture Variational Autoencoder Networks

"""
import torch
import torch.nn.init as init
from torch import nn
from torch.nn import functional as F
from networks.Layers import *

# Inference Network
class InferenceNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim):
    super(InferenceNet, self).__init__()

    # q(y|x)
    self.inference_qyx = torch.nn.ModuleList([
        nn.Linear(x_dim, 200),
        nn.ReLU(),
        nn.Linear(200, 200),
        nn.ReLU(),
        GumbelSoftmax(200, y_dim)
    ])

    # q(z|y,x)
    self.inference_qzyx = torch.nn.ModuleList([
        nn.Linear(x_dim + y_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
        Gaussian(512, z_dim)
    ])

  # q(y|x)
  def qyx(self, x, temperature, hard):
    num_layers = len(self.inference_qyx)
    for i, layer in enumerate(self.inference_qyx):
      if i == num_layers - 1:
        #last layer is gumbel softmax
        x = layer(x, temperature, hard)
      else:
        x = layer(x)
    return x

  # q(z|x,y)
  def qzxy(self, x, y):
    concat = torch.cat((x, y), dim=1)  
    for layer in self.inference_qzyx:
      concat = layer(concat)
    return concat
  
  def forward(self, x, temperature=1.0, hard=0):
    #x = Flatten(x)
    #print(x.shape)

    # q(y|x)
    logits, prob, y = self.qyx(x, temperature, hard)
    
    # q(z|x,y)
    z_mu, z_var, z = self.qzxy(x, y)

    output = {'mean': z_mu, 'var': z_var, 'gaussian': z,
              'logits': logits, 'prob_cat': prob, 'categorical': y}
    return output


# Generative Network
class GenerativeNet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim, psd, var_x, version = '2'):
    super(GenerativeNet, self).__init__()

    self.version = version

    # p(z|y)
    self.z_mu_prior = nn.Linear(y_dim, z_dim)
    self.z_var_prior = nn.Linear(y_dim, z_dim)

    self.h_mu = nn.Linear(512, x_dim)
    self.h_var = nn.Linear(512, x_dim)

    if self.version == '2' or self.version == '3':
      self.x_mu = nn.Linear(512, x_dim)
      self.x_var = nn.Linear(512, x_dim)
    self.psd = psd
    self.var_x = var_x

    # p(H|z)
    self.generative_pxz = torch.nn.ModuleList([
        nn.Linear(z_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
    ])
    if self.version == '2':
      self.generative_pxz_2 = torch.nn.ModuleList([
        nn.Linear(z_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
      ])
    if self.version == '3':
      self.generative_px_zy = torch.nn.ModuleList([
        nn.Linear(z_dim + y_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
      ])
      self.generative_ph_zy = torch.nn.ModuleList([
        nn.Linear(z_dim + y_dim, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.ReLU(),
      ])

  # p(z|y)
  def pzy(self, y):
    z_mu_prior = self.z_mu_prior(y)
    z_var_prior = F.softplus(self.z_var_prior(y))
    return z_mu_prior, z_var_prior
  
  # p(H|z)
  def pxz(self, z):
    for layer in self.generative_pxz:
      z = layer(z)
    h_mean = self.h_mu(z)
    h_var = F.softplus(self.h_var(z)) + 1e-5
    return h_mean, h_var

  def pxz_2(self, z):
    for layer in self.generative_pxz_2:
      z = layer(z)
    x_mean = self.x_mu(z)
    x_var = F.softplus(self.x_var(z)) + 1e-5
    return x_mean, x_var

  def px_zy(self, z, y):
    t = torch.cat((z,y), 1)
    for layer in self.generative_px_zy:
      t = layer(t)
    h_mean = self.h_mu(t)
    h_var = F.softplus(self.h_var(t)) + 1e-5
    return h_mean, h_var

  def ph_zy(self, z, y):
    t = torch.cat((z,y), 1)
    for layer in self.generative_ph_zy:
      t = layer(t)
    x_mean = self.x_mu(t)
    x_var = F.softplus(self.x_var(t)) + 1e-5
    return x_mean, x_var

  def forward(self, z, y):
    # p(z|y)
    z_mu_prior, z_var_prior = self.pzy(y)

    if self.version == '1' or self.version == '2':
      # p(h|z)
      h_mean, h_var = self.pxz(z)
    elif self.version == '3':
      #p(h|z,y)
      h_mean, h_var = self.ph_zy(z, y)

    eps = torch.randn_like(h_var)
    h = h_mean + eps * torch.sqrt(h_var)

    if self.version == '2':
      x_mean, x_var = self.pxz_2(z)
    elif self.version == '3':
      x_mean, x_var = self.px_zy(z, y)


    if self.version == '2':
      eps = torch.randn_like(h_var)
      aux = torch.ones_like(h_var)
      x_sample = x_mean + eps*torch.sqrt(x_var*aux)
      x_rec_mean = h*self.psd.repeat(int(h_mean.shape[0]),1) + x_sample
      x_rec = x_rec_mean + eps*torch.sqrt(self.var_x*aux)
    elif self.version == '1':
      eps = torch.randn_like(h_var)
      aux = torch.ones_like(h_var)
      x_rec = h*self.psd.repeat(int(h_mean.shape[0]),1) + eps*torch.sqrt(self.var_x*aux)
    else:
      eps = torch.randn_like(h_var)
      aux = torch.ones_like(h_var)
      x_sample = x_mean + eps * torch.sqrt(x_var * aux)
      x_rec_mean = h*self.psd.repeat(int(h_mean.shape[0]),1) + x_sample
      x_rec = x_rec_mean + eps * torch.sqrt(self.var_x * aux)




    output = {'y_mean': z_mu_prior, 'y_var': z_var_prior, 'x_rec': x_rec}
    return output


# GMVAE Network
class GMVAENet(nn.Module):
  def __init__(self, x_dim, z_dim, y_dim, psd, var_x, version = '2'):
    super(GMVAENet, self).__init__()

    self.inference = InferenceNet(x_dim, z_dim, y_dim)
    self.generative = GenerativeNet(x_dim, z_dim, y_dim, psd, var_x, version)

    # weight initialization
    for m in self.modules():
      if type(m) == nn.Linear or type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias.data is not None:
          init.constant_(m.bias, 0) 

  def forward(self, x, temperature=1.0, hard=0):
    x = x.view(x.size(0), -1)
    out_inf = self.inference(x, temperature, hard)
    z, y = out_inf['gaussian'], out_inf['categorical']
    out_gen = self.generative(z, y)
    
    # merge output
    output = out_inf
    for key, value in out_gen.items():
      output[key] = value
    return output
