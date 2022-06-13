# Import PyTorch
import torch
from torch import nn
from torch.autograd import Variable
# Import custom modules
from .loss import GaussianKLLoss, MaximumMeanDiscrepancyLoss

class Latent_module(nn.Module):
    def __init__(self, d_model, d_latent, variational_mode, z_var, device):

        super(Latent_module, self).__init__()

        self.variational_mode = variational_mode
        self.z_var = z_var
        self.loss_lambda = 1
        self.device = device
        
        if self.variational_mode < 5:
            self.context_to_mu = nn.Linear(d_model, d_latent)
            self.context_to_logvar = nn.Linear(d_model, d_latent)
            self.z_to_context = nn.Linear(d_latent, d_model)

            self.kl_criterion = GaussianKLLoss()

        if self.variational_mode in [5,6]:
            self.context_to_latent = nn.Linear(d_model, d_latent)
            self.latent_to_context = nn.Linear(d_latent, d_model)

            self.mmd_criterion = MaximumMeanDiscrepancyLoss(device=self.device)

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
            self.mmd_criterion = MaximumMeanDiscrepancyLoss(device=self.device)
        
        if self.variational_mode == 9:
            self.content_latent_encoder = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=4, stride=4), # (batch_size, d_model, 60)
                nn.GELU(),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=5), # (batch_size, d_model, 10)
                nn.GELU(),
                nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=5, stride=1), # (batch_size, d_model, 1)
                nn.GELU()
            )

            self.style_latent_encoder = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=4, stride=4), # (batch_size, d_model, 60)
                nn.GELU(),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=5), # (batch_size, d_model, 10)
                nn.GELU(),
                nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=5, stride=1), # (batch_size, d_model, 1)
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

            self.mmd_criterion = MaximumMeanDiscrepancyLoss(device=self.device)
            self.content_similiarity_criterion = nn.CosineEmbeddingLoss() #nn.CosineSimilarity()
            self.style_similiarity_criterion = nn.CosineEmbeddingLoss() #nn.CosineSimilarity()
        
        if self.variational_mode == 10:
            self.content_latent_encoder = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=4, stride=4), # (batch_size, d_model, 60)
                nn.GELU(),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=5), # (batch_size, d_model, 10)
                nn.GELU(),
                nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=5, stride=1), # (batch_size, d_model, 1)
                nn.GELU()
            )

            self.style_latent_encoder = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=4, stride=4), # (batch_size, d_model, 60)
                nn.GELU(),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=5), # (batch_size, d_model, 10)
                nn.GELU(),
                nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=5, stride=1), # (batch_size, d_model, 1)
                nn.GELU()
            )

            self.content_latent_to_mu = nn.Linear(d_latent, d_latent)
            self.content_latent_to_logvar = nn.Linear(d_latent, d_latent)
            self.style_latent_to_mu = nn.Linear(d_latent, d_latent)
            self.style_latent_to_logvar = nn.Linear(d_latent, d_latent)

            self.content_latent_decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_latent, out_channels=d_model, kernel_size=1, stride=1),
                nn.GELU()
            )
            self.style_latent_decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_latent, out_channels=d_model, kernel_size=1, stride=1),
                nn.GELU()
            )

            self.kl_criterion = GaussianKLLoss()
            self.content_similiarity_criterion = nn.CosineEmbeddingLoss() #nn.CosineSimilarity()
            self.style_similiarity_criterion = nn.CosineEmbeddingLoss() #nn.CosineSimilarity()

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

    #===================================#
    #==============CNN+WAE==============#
    #==============GMM&SIM==============#

        if self.variational_mode == 9:
            # Source sentence latent mapping
            encoder_out_src = encoder_out_src.permute(1, 2, 0) # From: (seq_len, batch_size, d_model)
            encoder_out_trg = encoder_out_trg.permute(1, 2, 0) # To: (batch_size, d_model, seq_len)

            # 1-1. Get content latent
            src_content_latent = self.content_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            trg_content_latent = self.content_latent_encoder(encoder_out_trg) # (batch_size, d_latent, 1)
            src_content_latent = src_content_latent.squeeze(2) # (batch_size, d_latent)
            trg_content_latent = trg_content_latent.squeeze(2) # (batch_size, d_latent)

            # 1-2. WAE Process
            # 1-2-1. Draw fake content latent from N~(0,2)
            fake_content_latent = torch.randn_like(src_content_latent).to(self.device) * 2 # (batch_size, d_latent)

            # 1-2-2. get mmd_loss between src_content_latent and fake_content_latent, trg_content_latent and fake_content_latent
            mmd_loss_content_src = self.mmd_criterion(src_content_latent, fake_content_latent, 2) # z_var is 2 now
            mmd_loss_content_trg = self.mmd_criterion(trg_content_latent, fake_content_latent, 2) # z_var is 2 now

            # 1-3. Similarity Loss - src_content_latent and trg_content_latent
            sim_loss_content = self.content_similiarity_criterion(src_content_latent, trg_content_latent, torch.Tensor([1]).to(self.device)) #self.content_similiarity_criterion(src_content_latent, trg_content_latent).mean()

            # 2-1. Get style latent
            src_style_latent = self.style_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            trg_style_latent = self.style_latent_encoder(encoder_out_trg) # (batch_size, d_latent, 1)
            src_style_latent = src_style_latent.squeeze(2) # (batch_size, d_latent)
            trg_style_latent = trg_style_latent.squeeze(2) # (batch_size, d_latent)

            # 2-2. WAE Process
            # 2-2-1. Draw fake style latent from predefined GMM distribution
            fake_style_latent = self.style_latent_gmm.sample((src_style_latent.shape[0],)) # (batch_size, d_latent)
            fake_style_latent = fake_style_latent.to(self.device)

            # 2-2-2. get mmd_loss between src_style_latent and fake_style_latent, trg_style_latent and fake_style_latent
            mmd_loss_style_src = self.mmd_criterion(src_style_latent, fake_style_latent, 0.5) # z_var is 0.5 now 
            mmd_loss_style_trg = self.mmd_criterion(trg_style_latent, fake_style_latent, 0.5) # Mixture dist에서 MMD가 제대로 작동하는지?

            # 2-3. Similarity Loss - src_style_latent and trg_style_latent
            sim_loss_style = self.style_similiarity_criterion(src_style_latent, trg_style_latent, torch.Tensor([-1]).to(self.device)) #self.style_similiarity_criterion(src_style_latent, trg_style_latent).mean()
            
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

        if self.variational_mode == 10:
            # Source sentence latent mapping
            encoder_out_src = encoder_out_src.permute(1, 2, 0) # From: (seq_len, batch_size, d_model)
            encoder_out_trg = encoder_out_trg.permute(1, 2, 0) # To: (batch_size, d_model, seq_len)

            # 1-1. Get content latent
            src_content_latent = self.content_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            trg_content_latent = self.content_latent_encoder(encoder_out_trg) # (batch_size, d_latent, 1)
            src_content_latent = src_content_latent.squeeze(2) # (batch_size, d_latent)
            trg_content_latent = trg_content_latent.squeeze(2) # (batch_size, d_latent)

            # 1-2. VAE Process
            # 1-2-1. Get mu and logvar from src_content_latent and trg_content_latent
            src_content_mu = self.content_latent_to_mu(src_content_latent) # (batch_size, d_latent)
            src_content_logvar = self.content_latent_to_logvar(src_content_latent) # (batch_size, d_latent)
            trg_content_mu = self.content_latent_to_mu(trg_content_latent) # (batch_size, d_latent)
            trg_content_logvar = self.content_latent_to_logvar(trg_content_latent) # (batch_size, d_latent)

            # 1-2-2. Reparameterization trick
            src_content_std = src_content_logvar.mul(0.5).exp_()
            src_content_eps = torch.randn_like(src_content_std).to(self.device)
            src_content_z = src_content_eps * src_content_std + src_content_mu # (batch_size, d_latent)

            trg_content_std = trg_content_logvar.mul(0.5).exp_()
            trg_content_eps = torch.randn_like(trg_content_std).to(self.device)
            trg_content_z = trg_content_eps * trg_content_std + trg_content_mu # (batch_size, d_latent)

            # 1-2-3. Get kl loss with N~(0, 1)
            kl_loss_content_src = self.kl_criterion(src_content_mu, src_content_logvar,
                                                    torch.Tensor([0]).to(self.device),
                                                    torch.Tensor([1]).to(self.device))
            kl_loss_content_trg = self.kl_criterion(trg_content_mu, trg_content_logvar,
                                                    torch.Tensor([0]).to(self.device),
                                                    torch.Tensor([1]).to(self.device))

            # 1-3. Similarity Loss - src_content_latent and trg_content_latent
            sim_loss_content = self.content_similiarity_criterion(src_content_z, trg_content_z, torch.Tensor([1]).to(self.device))

            # 2-1. Get style latent
            src_style_latent = self.style_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            trg_style_latent = self.style_latent_encoder(encoder_out_trg) # (batch_size, d_latent, 1)
            src_style_latent = src_style_latent.squeeze(2) # (batch_size, d_latent)
            trg_style_latent = trg_style_latent.squeeze(2) # (batch_size, d_latent)

            # 2-2. VAE Process
            # 2-2-1. Get mu and logvar from src_style_latent and trg_style_latent
            src_style_mu = self.style_latent_to_mu(src_style_latent) # (batch_size, d_latent)
            src_style_logvar = self.style_latent_to_logvar(src_style_latent) # (batch_size, d_latent)
            trg_style_mu = self.style_latent_to_mu(trg_style_latent) # (batch_size, d_latent)
            trg_style_logvar = self.style_latent_to_logvar(trg_style_latent) # (batch_size, d_latent)

            # 2-2-2. Reparameterization trick
            src_style_std = src_style_logvar.mul(0.5).exp_()
            src_style_eps = torch.randn_like(src_style_std).to(self.device)
            src_style_z = src_style_eps * src_style_std + src_style_mu # (batch_size, d_latent)

            trg_style_std = trg_style_logvar.mul(0.5).exp_()
            trg_style_eps = torch.randn_like(trg_style_std).to(self.device)
            trg_style_z = trg_style_eps * trg_style_std + trg_style_mu # (batch_size, d_latent)

            # 2-2-3. Get kl loss with N~(0, 1)
            kl_loss_style_src = self.kl_criterion(src_style_mu, src_style_logvar, 
                                                  torch.Tensor([0]).to(self.device),
                                                  torch.Tensor([1]).to(self.device))
            kl_loss_style_trg = self.kl_criterion(trg_style_mu, trg_style_logvar,
                                                  torch.Tensor([0]).to(self.device),
                                                  torch.Tensor([1]).to(self.device))

            # 2-3. Similarity Loss - src_style_latent and trg_style_latent
            sim_loss_style = self.style_similiarity_criterion(src_style_z, trg_style_z, torch.Tensor([-1]).to(self.device))

            # 3-1. Translate each src latent to d_model dimension
            src_content_latent = self.content_latent_decoder(src_content_z.unsqueeze(2)) # (batch_size, d_model, 1)
            src_style_latent = self.style_latent_decoder(src_style_z.unsqueeze(2)) # (batch_size, d_model, 1)

            # 3-2. add each src latent and repeat
            src_latent = src_content_latent + src_style_latent # (batch_size, d_model, 1)
            src_latent = src_latent.repeat(1, 1, encoder_out_src.size(2)) # (batch_size, d_model, seq_len)

            # 4. Define dist_loss
            dist_loss = kl_loss_content_src + kl_loss_content_trg + kl_loss_style_src + kl_loss_style_trg
            sim_loss = sim_loss_content + sim_loss_style # maximize sim_loss_content, minimize sim_loss_style
            
            dist_loss = dist_loss + sim_loss

            # 5. Get output
            encoder_out_total = torch.add(encoder_out_src, src_latent)
            encoder_out_total = encoder_out_total.permute(2, 0, 1) # (seq_len, batch_size, d_model)

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
        
        if self.variational_mode == 9:
            # Source sentence latent mapping
            encoder_out_src = encoder_out_src.permute(1, 2, 0) # From: (seq_len, batch_size, d_model)

            # 1-1. Get content latent
            src_content_latent = self.content_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            src_content_latent = src_content_latent.squeeze(2) # (batch_size, d_latent)

            # 2-1. Get style latent
            src_style_latent = self.style_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            src_style_latent = src_style_latent.squeeze(2) # (batch_size, d_latent)
            
            # 3-1. Translate each src latent to d_model dimension
            src_content_latent = self.content_latent_decoder(src_content_latent.unsqueeze(2)) # (batch_size, d_model, 1)
            src_style_latent = self.style_latent_decoder(src_style_latent.unsqueeze(2)) # (batch_size, d_model, 1)

            # 3-2. add each src latent and repeat
            src_latent = src_content_latent + src_style_latent # (batch_size, d_model, 1)
            src_latent = src_latent.repeat(1, 1, encoder_out_src.size(2)) # (batch_size, d_model, seq_len)

            # 5. Get output
            encoder_out_total = torch.add(encoder_out_src, src_latent)
            encoder_out_total = encoder_out_total.permute(2, 0, 1) # (seq_len, batch_size, d_model)

        if self.variational_mode == 10:
            # Source sentence latent mapping
            encoder_out_src = encoder_out_src.permute(1, 2, 0) # From: (seq_len, batch_size, d_model)

            # 1-1. Get content latent
            src_content_latent = self.content_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            src_content_latent = src_content_latent.squeeze(2) # (batch_size, d_latent)

            # 1-2. VAE Process
            # 1-2-1. Get mu and logvar from src_content_latent and trg_content_latent
            src_content_mu = self.content_latent_to_mu(src_content_latent) # (batch_size, d_latent)
            src_content_logvar = self.content_latent_to_logvar(src_content_latent) # (batch_size, d_latent)

            # 1-2-2. Reparameterization trick
            src_content_std = src_content_logvar.mul(0.5).exp_()
            src_content_eps = torch.randn_like(src_content_std).to(self.device)
            src_content_z = src_content_eps * src_content_std + src_content_mu # (batch_size, d_latent)

            # 2-1. Get style latent
            src_style_latent = self.style_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            src_style_latent = src_style_latent.squeeze(2) # (batch_size, d_latent)

            # 2-2. VAE Process
            # 2-2-1. Get mu and logvar from src_style_latent and trg_style_latent
            src_style_mu = self.style_latent_to_mu(src_style_latent) # (batch_size, d_latent)
            src_style_logvar = self.style_latent_to_logvar(src_style_latent) # (batch_size, d_latent)

            # 2-2-2. Reparameterization trick
            src_style_std = src_style_logvar.mul(0.5).exp_()
            src_style_eps = torch.randn_like(src_style_std).to(self.device)
            src_style_z = src_style_eps * src_style_std + src_style_mu # (batch_size, d_latent)

            # 3-1. Translate each src latent to d_model dimension
            src_content_latent = self.content_latent_decoder(src_content_z.unsqueeze(2)) # (batch_size, d_model, 1)
            src_style_latent = self.style_latent_decoder(src_style_z.unsqueeze(2)) # (batch_size, d_model, 1)

            # 3-2. add each src latent and repeat
            src_latent = src_content_latent + src_style_latent # (batch_size, d_model, 1)
            src_latent = src_latent.repeat(1, 1, encoder_out_src.size(2)) # (batch_size, d_model, seq_len)

            # 5. Get output
            encoder_out_total = torch.add(encoder_out_src, src_latent)
            encoder_out_total = encoder_out_total.permute(2, 0, 1) # (seq_len, batch_size, d_model)

        return encoder_out_total
