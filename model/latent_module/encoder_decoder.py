import torch
import torch.nn as nn

class latent_encoder_module(nn.Module):
    def __init__(self, d_model, d_latent, token_length, mode=1):

        super(latent_encoder_module, self).__init__()
        if mode == 1:
            if token_length == 100:
                self.encoder = nn.Sequential(
                            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=3), # (batch_size, d_model, 33)
                            nn.GELU(),
                            nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=3, stride=3), # (batch_size, d_model, 11)
                            nn.GELU(),
                            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=3), # (batch_size, d_model, 3)
                            nn.GELU(),
                            nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=3, stride=3), # (batch_size, d_model, 1)
                            nn.GELU()
                        )
            if token_length == 200:
                self.encoder = nn.Sequential(
                            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=3), # (batch_size, d_model, 66)
                            nn.GELU(),
                            nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=3, stride=3), # (batch_size, d_model, 22)
                            nn.GELU(),
                            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=3), # (batch_size, d_model, 7)
                            nn.GELU(),
                            nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=3, stride=3), # (batch_size, d_model, 2)
                            nn.GELU(),
                            nn.Conv1d(in_channels=d_latent, out_channels=d_latent, kernel_size=2, stride=3), # (batch_size, d_model, 1)
                            nn.GELU()
                        )
            if token_length == 300:
                self.encoder = nn.Sequential(
                            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=3), # (batch_size, d_model, 100)
                            nn.GELU(),
                            nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=3, stride=3), # (batch_size, d_model, 33)
                            nn.GELU(),
                            nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=3), # (batch_size, d_model, 11)
                            nn.GELU(),
                            nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=3, stride=3), # (batch_size, d_model, 3)
                            nn.GELU(),
                            nn.Conv1d(in_channels=d_latent, out_channels=d_latent, kernel_size=3, stride=3), # (batch_size, d_model, 1)
                            nn.GELU()
                        )
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=d_latent, kernel_size=3, stride=1, bias=True),
                nn.ReLU(inplace=False)
            )

    def forward(self, src):
        latent = self.encoder(src) # (batch, d_latent, seq_len)
        if self.mode != 1:
            latent, _ = torch.max(latent, dim=2)
            latent = latent.unsqueeze(2)
        
        return latent

class latent_decoder_module(nn.Module):
    def __init__(self, d_model, d_latent, token_length, mode=1):

        super(latent_decoder_module, self).__init__()
        self.token_length = token_length
        if mode == 1:
            if token_length == 100:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose1d(in_channels=d_latent, out_channels=256, kernel_size=10, stride=1),
                    nn.GELU(),
                    nn.ConvTranspose1d(in_channels=256, out_channels=512, kernel_size=5, stride=3, output_padding=1),
                    nn.GELU(),
                    nn.ConvTranspose1d(in_channels=512, out_channels=d_model, kernel_size=3, stride=3, output_padding=1),
                    nn.GELU(),
                )
            if token_length == 200:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose1d(in_channels=d_latent, out_channels=256, kernel_size=10, stride=1),
                    nn.GELU(),
                    nn.ConvTranspose1d(in_channels=256, out_channels=512, kernel_size=5, stride=3, output_padding=1),
                    nn.GELU(),
                    nn.ConvTranspose1d(in_channels=512, out_channels=d_model, kernel_size=3, stride=3, output_padding=1),
                    nn.GELU(),
                    nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=2, stride=2),
                    nn.GELU()
                )
            if token_length == 300:
                self.decoder = nn.Sequential(
                    nn.ConvTranspose1d(in_channels=d_latent, out_channels=256, kernel_size=10, stride=1),
                    nn.GELU(),
                    nn.ConvTranspose1d(in_channels=256, out_channels=512, kernel_size=5, stride=3, output_padding=1),
                    nn.GELU(),
                    nn.ConvTranspose1d(in_channels=512, out_channels=d_model, kernel_size=3, stride=3, output_padding=1),
                    nn.GELU(),
                    nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=3),
                    nn.GELU()
                )
        else:
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_latent, out_channels=d_model, kernel_size=1, stride=1),
                nn.ReLU(inplace=False)
            )

    def forward(self, src):
        latent = self.decoder(src) # (batch, d_latent, seq_len)
        if self.mode != 1:
            latent = latent.repeat(1, 1, self.token_length) # (batch_size, d_model, seq_len)
        
        return latent