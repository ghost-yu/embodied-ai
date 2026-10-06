from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor

class SmallConvNet(nn.Module):
    def __init__(self, feat_dim, last_nl=None, layernormalize=False, batchnorm=False):
        super(SmallConvNet, self).__init__()
        
        self.feat_dim = feat_dim
        self.last_nl = last_nl
        self.layernormalize = layernormalize
        self.batchnorm = batchnorm

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)
        self.fc = nn.Linear(1024, feat_dim)  # Assuming input images are 84x84

        if self.batchnorm:
            self.bn1 = nn.BatchNorm2d(32)
            self.bn2 = nn.BatchNorm2d(64)
            self.bn3 = nn.BatchNorm2d(64)
            self.bn_fc = nn.BatchNorm1d(feat_dim)

        if self.layernormalize:
            self.ln = nn.LayerNorm(feat_dim)

    def forward(self, x):
        t = x.size(2)
        enable_5_dims = False
        if x.dim() == 5:
            enable_5_dims = True
            x = rearrange(x, 'b c t h w -> (b t) c h w')
        if self.batchnorm:
            x = self.bn1(F.relu(self.conv1(x)))
            x = self.bn2(F.relu(self.conv2(x)))
            x = self.bn3(F.relu(self.conv3(x)))
        else:
            x = F.leaky_relu(self.conv1(x), negative_slope=0.01)
            x = F.leaky_relu(self.conv2(x), negative_slope=0.01)
            x = F.leaky_relu(self.conv3(x), negative_slope=0.01)

        x = x.reshape(x.size(0), -1)
        x = self.fc(x)

        if self.last_nl:
            x = self.last_nl(x)

        if self.layernormalize:
            x = self.ln(x)

        if enable_5_dims:
            x = rearrange(x, '(b t) c -> b t c', t=t)
        return x
    

 

class SmallDeconvNet(nn.Module):
    def __init__(self, 
                 latent_dim, 
                 out_channels, 
                 positional_bias=False, 
                 negative_slope=0.2):
        super(SmallDeconvNet, self).__init__()
        
        self.positional_bias = positional_bias
        self.negative_slope = negative_slope
        
        # 1) Fully-connected layer to go from latent_dim -> 8*8*64
        self.fc = nn.Linear(latent_dim, 8*8*64)
        
        # 2) First ConvTranspose2d:
        #    TF "same" with kernel_size=4, strides=2 from (8x8) -> (16x16)
        #    => padding=1 in PyTorch
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=4, 
            stride=2, 
            padding=1
        )
        
        # 3) Second ConvTranspose2d:
        #    TF "same" with kernel_size=8, strides=2 from (16x16) -> (32x32)
        #    => padding=3 in PyTorch
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=128, 
            out_channels=64, 
            kernel_size=8, 
            stride=2, 
            padding=3
        )
        
        # 4) Third ConvTranspose2d:
        #    TF "same" with kernel_size=8, strides=3 from (32x32) -> (96x96)
        #    There's no single integer padding that achieves exactly 96x96 
        #    unless we use output_padding=1. So we use:
        #    padding=3, output_padding=1.
        self.deconv3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=out_channels,
            kernel_size=8,
            stride=2,
            padding=3,
            
        )

        self.pos_bias_param = nn.Parameter(torch.zeros(1, 64,64), requires_grad=True)

    def forward(self, z):
        """
        z: (batch_size, latent_dim)  -- same as in TF code.
        Return shape will be (batch_size, out_channels, 84, 84).
        """
        # FC -> shape: (batch, 8*8*64)
        x = self.fc(z)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        
        # Reshape to (batch, channels=64, height=8, width=8)
        x = x.view(-1, 64, 8, 8)
        
        # Deconv 1 -> (batch, 128, 16, 16)
        x = self.deconv1(x)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        assert x.shape[2:] == (16, 16), f"Expected (16,16), got {x.shape[2:]}"
        
        # Deconv 2 -> (batch, 64, 32, 32)
        x = self.deconv2(x)
        x = F.leaky_relu(x, negative_slope=self.negative_slope)
        assert x.shape[2:] == (32, 32), f"Expected (32,32), got {x.shape[2:]}"
        
        # Deconv 3 -> (batch, out_channels, 96, 96)
        x = self.deconv3(x)
        # assert x.shape[2:] == (96, 96), f"Expected (96,96), got {x.shape[2:]}"
        
        # Crop to (84, 84)
        # x = x[:, :, 6:-6, 6:-6]
        # assert x.shape[2:] == (84, 84), f"Expected (84,84), got {x.shape[2:]}"
        
        # Optional positional bias
        if self.positional_bias:
            x = x + self.pos_bias_param
        
        return x


class VanillaVAE(nn.Module):


    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 hidden_dims: List = None,
                 **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 64] #128, 256, 512]

        # Build Encoder
        # for h_dim in hidden_dims:
        #     modules.append(
        #         nn.Sequential(
        #             nn.Conv2d(in_channels, out_channels=h_dim,
        #                       kernel_size= 3, stride= 2, padding  = 1),
        #             nn.BatchNorm2d(h_dim),
        #             nn.LeakyReLU())
        #     )
        #     in_channels = h_dim

        self.encoder = SmallConvNet(feat_dim=latent_dim*2, last_nl=None, layernormalize=False) #nn.Sequential(*modules)
        # self.fc_mu = nn.Linear(hidden_dims[-1]*4, latent_dim)
        # self.fc_var = nn.Linear(hidden_dims[-1]*4, latent_dim)


        # Build Decoder
        # modules = []

        # self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * 4)

        # hidden_dims.reverse()

        # for i in range(len(hidden_dims) - 1):
        #     modules.append(
        #         nn.Sequential(
        #             nn.ConvTranspose2d(hidden_dims[i],
        #                                hidden_dims[i + 1],
        #                                kernel_size=3,
        #                                stride = 2,
        #                                padding=1,
        #                                output_padding=1),
        #             nn.BatchNorm2d(hidden_dims[i + 1]),
        #             nn.LeakyReLU())
        #     )


        self.decoder = SmallDeconvNet(512, out_channels=3, positional_bias=True) #nn.Sequential(*modules)

        # self.final_layer = nn.Sequential(
        #                     nn.ConvTranspose2d(hidden_dims[-1],
        #                                        hidden_dims[-1],
        #                                        kernel_size=3,
        #                                        stride=2,
        #                                        padding=1,
        #                                        output_padding=1),
        #                     nn.BatchNorm2d(hidden_dims[-1]),
        #                     nn.LeakyReLU(),
        #                     nn.Conv2d(hidden_dims[-1], out_channels= 3,
        #                               kernel_size= 3, padding= 1),
        #                     nn.Tanh())

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu, log_var = torch.chunk(result, 2, dim=1)
        

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        result = z #self.decoder_input(z)
        result = self.decoder(result)
        
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), input, mu, log_var]

    def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss, 'Reconstruction_Loss':recons_loss.detach(), 'KLD':-kld_loss.detach()}

    def sample(self,
               num_samples:int,
               current_device: int, **kwargs) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
