# Import PyTorch
import torch
from torch import nn
from torch.autograd import Variable
# Import custom modules
from .loss import GaussianKLLoss, MaximumMeanDiscrepancyLoss

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

class Latent_module(nn.Module):
    def __init__(self, d_model, d_latent, variational_mode):

        super(Latent_module, self).__init__()

        self.variational_mode = variational_mode
        
        if self.variational_mode < 5:
            self.context_to_mu = nn.Linear(d_model, d_latent)
            self.context_to_logvar = nn.Linear(d_model, d_latent)
            self.z_to_context = nn.Linear(d_latent, d_model)

            self.kl_criterion = GaussianKLLoss()

        if self.variational_mode == 5:
            self.context_to_latent = nn.Linear(d_model, d_latent)
            self.latent_to_context = nn.Linear(d_latent, d_model)

            self.mmd_criterion = MaximumMeanDiscrepancyLoss()

        if self.variational_mode >= 6:
            self.latent_encoder = nn.Sequential(
                nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=5, stride=3),
                nn.GELU(),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=3),
                nn.GELU(),
                nn.Conv1d(in_channels=256, out_channels=128, kernel_size=10, stride=1),
                nn.GELU(),
            )

            self.context_to_mu = nn.Linear(128, 128)
            self.context_to_logvar = nn.Linear(128, 128)

            self.latent_decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=128, out_channels=256, kernel_size=10, stride=1),
                nn.GELU(),
                nn.ConvTranspose1d(in_channels=256, out_channels=512, kernel_size=5, stride=3),
                nn.GELU(),
                nn.ConvTranspose1d(in_channels=512, out_channels=1024, kernel_size=7, stride=3),
                nn.GELU(),
            )
            
            self.kl_criterion = GaussianKLLoss()
            self.mmd_criterion = MaximumMeanDiscrepancyLoss()
        
        if self.variational_mode == 8:
            self.content_latent_encoder = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=5, stride=5), # (batch_size, d_model, 60)
                nn.GELU(),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, stride=6), # (batch_size, d_model, 10)
                nn.GELU(),
                nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=10, stride=1), # (batch_size, d_model, 1)
                nn.GELU()
            )

            self.style_latent_encoder = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=5, stride=5), # (batch_size, d_model, 60)
                nn.GELU(),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size=5, stride=6), # (batch_size, d_model, 10)
                nn.GELU(),
                nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=10, stride=1), # (batch_size, d_model, 1)
                nn.GELU()
            )

            self.content_latent_decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_latent, out_channels=d_model, kernel_size=1, stride=1),
                nn.GELU()
            )
            self.style_latent_decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_latent, out_channels=d_model, kernel_size=1, stride=1),
                nn.GELU()
            )

            # Define Gaussian Mixture
            mix = torch.distributions.Categorical(torch.ones(2))
            mean_1 = torch.zeros(d_latent) - 2 # (d_latent)
            mean_2 = torch.zeros(d_latent) + 2 # (d_latent)
            sigma_1 = torch.ones(d_latent) * 0.5 # (d_latent)
            sigma_2 = torch.ones(d_latent) * 0.5 # (d_latent)
            comp = torch.distributions.Independent(torch.distributions.Normal(torch.stack([mean_1, mean_2]), torch.stack([sigma_1, sigma_2])), 1) # (2, d_latent)
            self.style_latent_gmm = torch.distributions.MixtureSameFamily(mix, comp) # 

            self.mmd_criterion = MaximumMeanDiscrepancyLoss()
            self.content_similiarity_criterion = nn.CosineEmbeddingLoss() #nn.CosineSimilarity()
            self.style_similiarity_criterion = nn.CosineEmbeddingLoss() #nn.CosineSimilarity()

    def forward(self, encoder_out_src, encoder_out_trg=None):

    #===================================#
    #===SRC|TRG -> Z+Encoder_out(Sum)===#
    #===================================#

        if self.variational_mode == 1:
            src_mu = self.context_to_mu(encoder_out_src) # (token, batch, d_latent)
            src_logvar = self.context_to_logvar(encoder_out_src) # (token, batch, d_latent)

            trg_mu = self.context_to_mu(encoder_out_trg) # (token, batch, d_latent)
            trg_logvar = self.context_to_logvar(encoder_out_trg) # (token, batch, d_latent)

            dist_loss = self.kl_criterion(src_mu.mean(dim=1), src_logvar.mean(dim=1), 
                                          trg_mu.mean(dim=1), trg_logvar.mean(dim=1)) # 

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
    #================WAE================#
    #===================================#

        if self.variational_mode == 5:
            # Source sentence latent mapping
            src_latent = self.context_to_latent(encoder_out_src) # (token, batch, d_latent)
            trg_latent = self.context_to_latent(encoder_out_trg) # (token, batch, d_latent)

            dist_loss = self.mmd_criterion(src_latent.mean(dim=1), trg_latent.mean(dim=1), 100) # z_var is 2 now

            #
            src_latent = self.latent_to_context(src_latent)

            encoder_out_total = torch.add(encoder_out_src, src_latent)

    #===================================#
    #==============CNN+VAE==============#
    #===================================#

        if self.variational_mode == 6:
            # Source sentence latent mapping
            encoder_out_src = encoder_out_src.transpose(1,2)
            encoder_out_trg = encoder_out_trg.transpose(1,2)

            src_latent = self.latent_encoder(encoder_out_src)
            trg_latent = self.latent_encoder(encoder_out_trg)

            src_mu = self.context_to_mu(src_latent.squeeze(2)) # (token, batch, d_latent)
            src_logvar = self.context_to_logvar(src_latent.squeeze(2)) # (token, batch, d_latent)

            trg_mu = self.context_to_mu(trg_latent.squeeze(2)) # (token, batch, d_latent)
            trg_logvar = self.context_to_logvar(trg_latent.squeeze(2)) # (token, batch, d_latent)
            
            dist_loss = self.kl_criterion(src_mu, src_logvar, trg_mu, trg_logvar) # 

            #
            src_latent = self.latent_decoder(src_latent)

            src_latent = src_latent.transpose(1,2)
            encoder_out_src = encoder_out_src.transpose(1,2)

            encoder_out_total = torch.add(encoder_out_src, src_latent)


    #===================================#
    #==============CNN+WAE==============#
    #===================================#

        if self.variational_mode == 7:
            # Source sentence latent mapping
            encoder_out_src = encoder_out_src.transpose(1,2)
            encoder_out_trg = encoder_out_trg.transpose(1,2)

            src_latent = self.latent_encoder(encoder_out_src)
            trg_latent = self.latent_encoder(encoder_out_trg)

            dist_loss = self.mmd_criterion(src_latent.squeeze(2), trg_latent.squeeze(2), 100) # z_var is 2 now

            #
            src_latent = self.latent_decoder(src_latent)

            src_latent = src_latent.transpose(1,2)
            encoder_out_src = encoder_out_src.transpose(1,2)

            encoder_out_total = torch.add(encoder_out_src, src_latent)

    #===================================#
    #==============CNN+WAE==============#
    #==============GMM&SIM==============#

        if self.variational_mode == 8:
            # Source sentence latent mapping
            encoder_out_src = encoder_out_src.permute(1, 2, 0) # From: (seq_len, batch_size, d_model)
            encoder_out_trg = encoder_out_trg.permute(1, 2, 0) # To: (batch_size, d_model, seq_len)

            # 1-1. Get content latent
            src_content_latent = self.content_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            trg_content_latent = self.content_latent_encoder(encoder_out_trg) # (batch_size, d_latent, 1)
            src_content_latent = src_content_latent.squeeze(2) # (batch_size, d_latent)
            trg_content_latent = trg_content_latent.squeeze(2) # (batch_size, d_latent)

            # 1-2. WAE Process
            # 1-2-1. Draw fake content latent from N~(0,1)
            fake_content_latent = torch.randn_like(src_content_latent) * 2 # (batch_size, d_latent)

            # 1-2-2. get mmd_loss between src_content_latent and fake_content_latent, trg_content_latent and fake_content_latent
            mmd_loss_content_src = self.mmd_criterion(src_content_latent, fake_content_latent, 2) # z_var is 2 now
            mmd_loss_content_trg = self.mmd_criterion(trg_content_latent, fake_content_latent, 2) # z_var is 2 now

            # 1-3. Similarity Loss - src_content_latent and trg_content_latent
            sim_loss_content = self.content_similiarity_criterion(src_content_latent, trg_content_latent, torch.Tensor([1]).to(device)) #self.content_similiarity_criterion(src_content_latent, trg_content_latent).mean()

            # 2-1. Get style latent
            src_style_latent = self.style_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            trg_style_latent = self.style_latent_encoder(encoder_out_trg) # (batch_size, d_latent, 1)
            src_style_latent = src_style_latent.squeeze(2) # (batch_size, d_latent)
            trg_style_latent = trg_style_latent.squeeze(2) # (batch_size, d_latent)

            # 2-2. WAE Process
            # 2-2-1. Draw fake style latent from predefined GMM distribution
            fake_style_latent = self.style_latent_gmm.sample((src_style_latent.shape[0],)) # (batch_size, d_latent)
            fake_style_latent = fake_style_latent.to(device)

            # 2-2-2. get mmd_loss between src_style_latent and fake_style_latent, trg_style_latent and fake_style_latent
            mmd_loss_style_src = self.mmd_criterion(src_style_latent, fake_style_latent, 0.5) # z_var is 1 now 
            mmd_loss_style_trg = self.mmd_criterion(trg_style_latent, fake_style_latent, 0.5) # Mixture dist에서 MMD가 제대로 작동하는지?

            # 2-3. Similarity Loss - src_style_latent and trg_style_latent
            sim_loss_style = self.style_similiarity_criterion(src_style_latent, trg_style_latent, torch.Tensor([-1]).to(device)) #self.style_similiarity_criterion(src_style_latent, trg_style_latent).mean()
            
            # 3-1. Translate each src latent to d_model dimension
            src_content_latent = self.content_latent_decoder(src_content_latent.unsqueeze(2)) # (batch_size, d_model, 1)
            src_style_latent = self.style_latent_decoder(src_style_latent.unsqueeze(2)) # (batch_size, d_model, 1)

            # 3-2. add each src latent and repeat
            src_latent = src_content_latent + src_style_latent # (batch_size, d_model, 1)
            src_latent = src_latent.repeat(1, 1, encoder_out_src.size(2)) # (batch_size, d_model, seq_len)

            # 4. Define dist_loss
            dist_loss = mmd_loss_content_src + mmd_loss_content_trg + mmd_loss_style_src + mmd_loss_style_trg
            sim_loss = sim_loss_content + sim_loss_style # maximize sim_loss_content, minimize sim_loss_style
            
            dist_loss = dist_loss + sim_loss

            # 5. Get output
            encoder_out_total = torch.add(encoder_out_src, src_latent)
            encoder_out_total = encoder_out_total.permute(2, 0, 1) # (seq_len, batch_size, d_model)

        return encoder_out_total, dist_loss
