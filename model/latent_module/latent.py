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
        self.loss_lambda = 1
        
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
                nn.Conv1d(in_channels=d_model, out_channels=d_latent, kernel_size=3, stride=1, bias=True),
                nn.ReLU(inplace=False)
            )

            self.context_to_mu = nn.Linear(d_latent, d_latent)
            self.context_to_logvar = nn.Linear(d_latent, d_latent)

            self.latent_gate = nn.Linear(d_latent, d_model)
            self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
            
            self.kl_criterion = GaussianKLLoss()
            self.mmd_criterion = MaximumMeanDiscrepancyLoss()

    def forward(self, encoder_out_src, encoder_out_trg=None):

    #===================================#
    #===SRC|TRG -> Z+Encoder_out(Sum)===#
    #===================================#

        if self.variational_mode == 1:
            src_mu = self.context_to_mu(encoder_out_src).mean(dim=0) # (token, batch, d_latent)
            src_logvar = self.context_to_logvar(encoder_out_src).mean(dim=0) # (token, batch, d_latent)

            trg_mu = self.context_to_mu(encoder_out_trg).mean(dim=0) # (token, batch, d_latent)
            trg_logvar = self.context_to_logvar(encoder_out_trg).mean(dim=0) # (token, batch, d_latent)

            dist_loss = self.kl_criterion(src_mu, src_logvar, 
                                          trg_mu, trg_logvar) # 

            # Re-parameterization
            std = src_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(src_mu)

            # resize_z = self.z_to_context(z)

            # encoder_out_total = torch.add(encoder_out_src, resize_z)
            # Augment semantically
            src_latent = self.z_to_context(z) # (batch, d_model)
            # gate = torch.sigmoid(src_latent + encoder_out_src) # (token, batch, d_model)

            gated_latent = 0.5 * src_latent
            gated_encoder_out = encoder_out_src * 0.5 # (token, batch, d_model)
            encoder_out_total = torch.add(gated_encoder_out, gated_latent)

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
            
            encoder_out_src = encoder_out_src.permute(1, 2, 0) # (batch, d_model, token)
            encoder_out_trg = encoder_out_trg.permute(1, 2, 0) # (batch, d_model, token)

            # Encoding
            src_latent = self.latent_encoder(encoder_out_src) # (batch, d_latent, token-k)
            trg_latent = self.latent_encoder(encoder_out_trg) # (batch, d_latent, token-k)

            src_latent, _ = torch.max(src_latent, dim=2) # (batch, d_latent)
            trg_latent, _ = torch.max(trg_latent, dim=2) # (batch, d_latent)

            src_mu = self.context_to_mu(src_latent) # (batch, d_latent)
            src_logvar = self.context_to_logvar(src_latent) # (batch, d_latent)

            trg_mu = self.context_to_mu(trg_latent) # (batch, d_latent)
            trg_logvar = self.context_to_logvar(trg_latent) # (batch, d_latent)
            
            dist_loss = self.kl_criterion(src_mu, src_logvar, trg_mu, trg_logvar)

            # Re-parameterization
            std = src_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(src_mu) # (batch, d_latent)

            # Augment semantically
            src_latent = self.latent_gate(z) # (batch, d_model)
            # gate = torch.sigmoid(src_latent.unsqueeze(2) + encoder_out_src)

            # encoder_out_total = (gate * src_latent.unsqueeze(2)) + ((1-gate) * encoder_out_src)
            # encoder_out_total = encoder_out_total.permute(2, 0, 1) # (token, batch, d_model)
            # encoder_out_total = (gate * src_latent.unsqueeze(2)) + ((1-gate) * encoder_out_src)
            encoder_out_src = encoder_out_src.permute(2, 0, 1)
            encoder_out_total = (0.5 * src_latent) + (0.5 * encoder_out_src)
            # encoder_out_total = encoder_out_total.permute(2, 0, 1) # (token, batch, d_model)

    #===================================#
    #==============CNN+WAE==============#
    #===================================#

        if self.variational_mode == 8:

            encoder_out_src = encoder_out_src.permute(1, 2, 0) # (batch, d_model, token)
            encoder_out_trg = encoder_out_trg.permute(1, 2, 0) # (batch, d_model, token)

            # Encoding
            src_latent = self.latent_encoder(encoder_out_src) # (batch, d_latent, token-k)
            trg_latent = self.latent_encoder(encoder_out_trg) # (batch, d_latent, token-k)

            src_latent, _ = src_latent.max(dim=2) # (batch, d_latent)
            trg_latent, _ = trg_latent.max(dim=2) # (batch, d_latent)

            # MMD loss
            dist_loss = self.mmd_criterion(src_latent, trg_latent, self.z_var)

            # Augment semantically
            src_latent = self.latent_gate(src_latent) # (batch, d_model)
            # gate = torch.sigmoid(src_latent.unsqueeze(2) + encoder_out_src)

            # encoder_out_total = (gate * src_latent.unsqueeze(2)) + ((1-gate) * encoder_out_src)
            encoder_out_src = encoder_out_src.permute(2, 0, 1)
            gated_latent = 0.5 * src_latent
            gated_encoder_out = encoder_out_src * 0.5
            encoder_out_total = torch.add(gated_encoder_out, gated_latent)
            # encoder_out_total = encoder_out_total.permute(2, 0, 1) # (token, batch, d_model)

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

        if self.variational_mode == 6:

            # Source sentence latent mapping
            src_latent = self.context_to_latent(encoder_out_src) # (token, batch, d_latent)
            src_latent = self.latent_to_context(src_latent)

            encoder_out_total = torch.add(encoder_out_src, src_latent)

    #===================================#
    #==============CNN+VAE==============#
    #===================================#

        if self.variational_mode == 7:

            encoder_out_src = encoder_out_src.transpose(0,1) # (batch, token, d_model)
            encoder_out_src = encoder_out_src.transpose(1,2) # (batch, d_model, token)

            src_latent = self.latent_encoder(encoder_out_src) # (token, batch, d_latent)

            src_mu = self.context_to_mu(src_latent.squeeze(2)) # (token, batch, d_latent)

            src_latent = self.latent_decoder(src_latent) # (batch, d_model, token)

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