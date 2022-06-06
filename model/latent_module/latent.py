# Import PyTorch
import torch
from torch import nn
from torch.autograd import Variable
# Import custom modules
from .loss import GaussianKLLoss, MaximumMeanDiscrepancyLoss

class Latent_module(nn.Module):
    def __init__(self, d_model, d_latent, variational_mode, z_var):

        super(Latent_module, self).__init__()

        self.variational_mode = variational_mode
        self.z_var = z_var
        self.loss_lambda = 100
        
        if self.variational_mode < 5:
            self.context_to_mu = nn.Linear(d_model, d_latent)
            self.context_to_logvar = nn.Linear(d_model, d_latent)
            self.z_to_context = nn.Linear(d_latent, d_model)

            self.kl_criterion = GaussianKLLoss()

        if self.variational_mode in [5,6]:
            self.context_to_latent = nn.Linear(d_model, d_latent)
            self.latent_to_context = nn.Linear(d_latent, d_model)

            self.mmd_criterion = MaximumMeanDiscrepancyLoss()

        if self.variational_mode >= 7:
            self.latent_encoder = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=5, stride=3, bias=False),
                nn.GELU(),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=3, bias=False),
                nn.GELU(),
                nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=10, stride=1, bias=False),
                nn.GELU(),
            )

            self.context_to_mu = nn.Linear(d_latent, d_latent)
            self.context_to_logvar = nn.Linear(d_latent, d_latent)

            self.latent_decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_latent, out_channels=256, kernel_size=10, stride=1, bias=False),
                nn.GELU(),
                nn.ConvTranspose1d(in_channels=256, out_channels=512, kernel_size=5, stride=3, bias=False),
                nn.GELU(),
                nn.ConvTranspose1d(in_channels=512, out_channels=d_model, kernel_size=7, stride=3, bias=False),
                nn.GELU(),
            )
            
            # Add for GMVAE
            self.num_cls = 2
            self.gm_context_to_cls = nn.Linear(300, self.num_cls) # (token, num_class)
            self.init_weight()
            self.gm_context_to_mu = nn.Linear(d_model + self.num_cls, d_latent)
            self.gm_context_to_logvar = nn.Linear(d_model + self.num_cls, d_latent)
            self.gm_z_to_context = nn.Linear(d_latent, d_model + self.num_cls)
       
            self.kl_criterion = GaussianKLLoss()
            self.mmd_criterion = MaximumMeanDiscrepancyLoss()

            
    def init_weight(self):
        initrange = 0.5
        self.gm_context_to_cls.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, encoder_out_src, encoder_out_trg=None):

    #===================================#
    #===SRC|TRG -> Z+Encoder_out(Sum)===#
    #===================================#

        if self.variational_mode == 1:
            src_mu = self.context_to_mu(encoder_out_src) # (token, batch, d_latent)
            src_logvar = self.context_to_logvar(encoder_out_src) # (token, batch, d_latent)

            trg_mu = self.context_to_mu(encoder_out_trg) # (token, batch, d_latent)
            trg_logvar = self.context_to_logvar(encoder_out_trg) # (token, batch, d_latent)

            dist_loss = self.kl_criterion(src_mu.mean(dim=0), src_logvar.mean(dim=0), 
                                          trg_mu.mean(dim=0), trg_logvar.mean(dim=0)) # 

            # Re-parameterization
            std = src_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(src_mu)

            resize_z = self.z_to_context(z)

            encoder_out_total = torch.add(encoder_out_src, resize_z)

    #===================================#
    #==SRC|TRG -> Z+Encoder_out(View)===#
    #===================================#

        if self.variational_mode == 2:
            batch_size = encoder_out_src.size(1)
            # Source sentence latent mapping
            src_mu = self.context_to_mu(encoder_out_src) # (token, batch, d_latent)
            src_logvar = self.context_to_logvar(encoder_out_src) # (token, batch, d_latent)

            trg_mu = self.context_to_mu(encoder_out_trg) # (token, batch, d_latent)
            trg_logvar = self.context_to_logvar(encoder_out_trg) # (token, batch, d_latent)

            mu1 = src_mu.view(batch_size, -1)
            logvar1 = src_logvar.view(batch_size, -1)
            mu2 = trg_mu.view(batch_size, -1)
            logvar2 = trg_logvar.view(batch_size, -1)

            numerator = logvar1.exp() + torch.pow(mu1 - mu2, 2)
            fraction = torch.div(numerator, (logvar2.exp()))

            dist_loss = 0.5 * torch.sum(logvar2 - logvar1 + fraction - 1, dim=0)
            dist_loss = dist_loss.mean()

            # Re-parameterization
            std = src_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(src_mu)

            resize_z = self.z_to_context(z)

            encoder_out_total = torch.add(encoder_out_src, resize_z)

    #===================================#
    #===========SRC -> Only Z===========#
    #===================================#

        if self.variational_mode == 3:
            # Source sentence latent mapping
            src_mu = self.context_to_mu(encoder_out_src) # (token, batch, d_latent)
            src_logvar = self.context_to_logvar(encoder_out_src) # (token, batch, d_latent)

            # KL Divergence
            mu = src_mu.view(encoder_out_src.size(1), -1)
            logvar = src_logvar.view(encoder_out_src.size(1), -1)
            dist_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # Re-parameterization
            std = src_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(src_mu)

            encoder_out_total = self.z_to_context(z)

    #===================================#
    #=========SRC|TRG -> Only Z=========#
    #===================================#

        if self.variational_mode == 4:
            # Source sentence latent mapping
            src_mu = self.context_to_mu(encoder_out_src) # (token, batch, d_latent)
            src_logvar = self.context_to_logvar(encoder_out_src) # (token, batch, d_latent)

            trg_mu = self.context_to_mu(encoder_out_trg) # (token, batch, d_latent)
            trg_logvar = self.context_to_logvar(encoder_out_trg) # (token, batch, d_latent)

            dist_loss = self.kl_criterion(src_mu, src_logvar, trg_mu, trg_logvar) # 

            # Re-parameterization
            std = src_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(src_mu)

            resize_z = self.z_to_context(z)

            # Re-parameterization
            std = src_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(src_mu)

            encoder_out_total = self.z_to_context(z)

    #===================================#
    #=============WAE(mean)=============#
    #===================================#

        if self.variational_mode == 5:
            # Source sentence latent mapping
            src_latent = self.context_to_latent(encoder_out_src) # (token, batch, d_latent)
            trg_latent = self.context_to_latent(encoder_out_trg) # (token, batch, d_latent)

            dist_loss = self.mmd_criterion(src_latent.mean(dim=0), 
                                           trg_latent.mean(dim=0), self.z_var) # z_var is 2 now

            #
            src_latent = self.latent_to_context(src_latent)

            encoder_out_total = torch.add(encoder_out_src, src_latent)

    #===================================#
    #=============WAE(view)=============#
    #===================================#

        if self.variational_mode == 6:

            batch_size = encoder_out_src.size(1)
            # Source sentence latent mapping
            src_latent = self.context_to_latent(encoder_out_src) # (token, batch, d_latent)
            trg_latent = self.context_to_latent(encoder_out_trg) # (token, batch, d_latent)

            dist_loss = self.mmd_criterion(src_latent.transpose(0,1).contiguous().view(batch_size, -1), 
                                           trg_latent.transpose(0,1).contiguous().view(batch_size, -1), self.z_var)

            #
            src_latent = self.latent_to_context(src_latent)

            encoder_out_total = torch.add(encoder_out_src, src_latent)

    #===================================#
    #==============CNN+VAE==============#
    #===================================#

        if self.variational_mode == 7:
            
            encoder_out_src = encoder_out_src.transpose(0,1) # (batch, token, d_model)
            encoder_out_trg = encoder_out_trg.transpose(0,1) # (batch, token, d_model)

            # Source sentence latent mapping
            encoder_out_src = encoder_out_src.transpose(1,2) # (batch, d_model, token)
            encoder_out_trg = encoder_out_trg.transpose(1,2) # (batch, d_model, token)

            src_latent = self.latent_encoder(encoder_out_src) # (batch, d_latent, 1)
            trg_latent = self.latent_encoder(encoder_out_trg) # (batch, d_latent, 1)

            src_mu = self.context_to_mu(src_latent.squeeze(2)) # (batch, d_latent)
            src_logvar = self.context_to_logvar(src_latent.squeeze(2)) # (batch, d_latent)

            trg_mu = self.context_to_mu(trg_latent.squeeze(2)) # (batch, d_latent)
            trg_logvar = self.context_to_logvar(trg_latent.squeeze(2)) # (batch, d_latent)
            
            dist_loss = self.kl_criterion(src_mu, src_logvar, trg_mu, trg_logvar)

            #
            src_latent = self.latent_decoder(src_latent) # (batch, d_model, token)
            src_latent = src_latent.transpose(1,2).transpose(0,1) # (token, batch, d_model)
            encoder_out_src = encoder_out_src.transpose(1,2).transpose(0,1) # (token, batch, d_model)
            encoder_out_total = torch.add(encoder_out_src, src_latent)


    #===================================#
    #==============CNN+WAE==============#
    #===================================#

        if self.variational_mode == 8:

            encoder_out_src = encoder_out_src.permute(1,2,0) # (batch, d_model, token)
            encoder_out_trg = encoder_out_trg.permute(1,2,0) # (batch, d_model, token)

            src_latent = self.latent_encoder(encoder_out_src) # (batch, d_latent, 1)
            trg_latent = self.latent_encoder(encoder_out_trg) # (batch, d_latent, 1)

            dist_loss = self.mmd_criterion(src_latent.squeeze(2), 
                                           trg_latent.squeeze(2), self.z_var)

            #
            src_latent = self.latent_decoder(src_latent) # (batch, d_model, token)

            src_latent = src_latent.permute(2, 0, 1) # (token, batch, d_model)
            encoder_out_src = encoder_out_src.permute(2, 0, 1) # (token, batch, d_model)

            encoder_out_total = torch.add(encoder_out_src, src_latent)

        return encoder_out_total, dist_loss * self.loss_lambda

    def generate(self, encoder_out_src):

    #===================================#
    #===SRC|TRG -> Z+Encoder_out(Sum)===#
    #===================================#

        if self.variational_mode == 1:
            src_mu = self.context_to_mu(encoder_out_src) # (token, batch, d_latent)
            resize_z = self.z_to_context(src_mu) # (token, batch, d_model)

            encoder_out_total = torch.add(encoder_out_src, resize_z)

    #===================================#
    #==SRC|TRG -> Z+Encoder_out(View)===#
    #===================================#

        if self.variational_mode == 2:

            batch_size = encoder_out_src.size(1)
            src_mu = self.context_to_mu(encoder_out_src) # (token, batch, d_latent)
            resize_z = self.z_to_context(src_mu) # (token, batch, d_model)

            encoder_out_total = torch.add(encoder_out_src, resize_z)

    #===================================#
    #===========SRC -> Only Z===========#
    #===================================#

        if self.variational_mode == 3:
            # Source sentence latent mapping
            src_mu = self.context_to_mu(encoder_out_src) # (token, batch, d_latent)
            encoder_out_total = self.z_to_context(src_mu)

    #===================================#
    #=========SRC|TRG -> Only Z=========#
    #===================================#

        # if self.variational_mode == 4:
        #     # Source sentence latent mapping
        #     src_mu = self.context_to_mu(encoder_out_src) # (token, batch, d_latent)
        #     src_logvar = self.context_to_logvar(encoder_out_src) # (token, batch, d_latent)

        #     trg_mu = self.context_to_mu(encoder_out_trg) # (token, batch, d_latent)
        #     trg_logvar = self.context_to_logvar(encoder_out_trg) # (token, batch, d_latent)

        #     dist_loss = self.kl_criterion(src_mu, src_logvar, trg_mu, trg_logvar) # 

        #     # Re-parameterization
        #     std = src_logvar.mul(0.5).exp_()
        #     eps = Variable(std.data.new(std.size()).normal_())
        #     z = eps.mul(std).add_(src_mu)

        #     resize_z = self.z_to_context(z)

        #     # Re-parameterization
        #     std = src_logvar.mul(0.5).exp_()
        #     eps = Variable(std.data.new(std.size()).normal_())
        #     z = eps.mul(std).add_(src_mu)

        #     encoder_out_total = self.z_to_context(z)

    #===================================#
    #================WAE================#
    #===================================#

        if self.variational_mode == 5:
            # Source sentence latent mapping
            src_latent = self.context_to_latent(encoder_out_src) # (token, batch, d_latent)
            src_latent = self.latent_to_context(src_latent) # (token, batch, d_model)

            encoder_out_total = torch.add(encoder_out_src, src_latent)

    #===================================#
    #==============CNN+VAE==============#
    #===================================#

        if self.variational_mode == 7:

            encoder_out_src = encoder_out_src.transpose(0,1) # (batch, token, d_model)
            encoder_out_src = encoder_out_src.transpose(1,2) # (batch, d_model, token)

            src_latent = self.latent_encoder(encoder_out_src) # (token, batch, d_latent)

            src_mu = self.context_to_mu(src_latent.squeeze(2)) # (token, batch, d_latent)

            src_latent = self.latent_decoder(src_mu) # (batch, d_model, token)

            src_latent = src_latent.transpose(1,2).transpose(0,1) # (token, batch, d_model)
            encoder_out_src = encoder_out_src.transpose(1,2).transpose(0,1) # (token, batch, d_model)

            encoder_out_total = torch.add(encoder_out_src, src_latent)
            
    #===================================#
    #==============CNN+WAE==============#
    #===================================#

        if self.variational_mode == 8:
            # Source sentence latent mapping
            encoder_out_src = encoder_out_src.transpose(1,2)

            src_latent = self.latent_encoder(encoder_out_src)

            src_latent = self.latent_decoder(src_latent) # (batch, d_model, token)

            src_latent = src_latent.transpose(1,2).transpose(0,1) # (token, batch, d_model)
            encoder_out_src = encoder_out_src.transpose(0,1) # (token, batch, d_model)

            encoder_out_total = torch.add(encoder_out_src, src_latent)

        return encoder_out_total
    
    
    #===================================#
    #================GMVAE==============#
    #===================================#

        if self.variational_mode == 11:

            # classifiction 
            x_to_cls = encoder_out_src.mean(dim=2).view(encoder_out_src.size(1), -1) # (token, batch, 1) 
            distribution_cls = self.gm_context_to_cls(x_to_cls) # (batch, num_cls) 
            distribution_cls_cp = distribution_cls.unsqueeze(1).repeat(1, encoder_out_src.size(0), 1).view(300, 48, -1) # (batch, token, num_cls)

            # concat
            gm_inp = torch.cat((encoder_out_src, distribution_cls_cp), dim=2) # (tokn, batch, d_latent + num_cls)

            src_mu = self.gm_context_to_mu(gm_inp) # (token, batch, d_latent)
            src_logvar = self.gm_context_to_logvar(gm_inp) # (token, batch, d_latent)

            mu = src_mu.view(gm_inp.size(1), -1)
            logvar = src_logvar.view(gm_inp.size(1), -1)
            dist_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            std = src_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(src_mu)

            return encoder_out_total, dist_loss, distribution_cls
    
