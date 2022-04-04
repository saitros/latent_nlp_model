import math
# Import PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
# Import Huggingface
from transformers import BartForConditionalGeneration, BartConfig
#
from ..loss import GaussianKLLoss

class Bart(nn.Module):
    def __init__(self, isPreTrain, variational, d_latent, emb_src_trg_weight_sharing=True):
        super().__init__()

        """
        Initialize WAE model
        
        Args:
            encoder_config (dictionary): encoder transformer's configuration
            d_latent (int): latent dimension size
            device (torch.device): 
        Returns:
            log_prob (torch.Tensor): log probability of each word 
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
            z (torch.Tensor): sampled latent vector
        """
        self.d_latent = d_latent
        self.isPreTrain = isPreTrain
        self.emb_src_trg_weight_sharing = emb_src_trg_weight_sharing
        self.model_config = BartConfig.from_pretrained("facebook/bart-base")
        self.pad_idx = self.model_config.pad_token_id

        if self.isPreTrain:
            self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
        else:
            self.model = BartForConditionalGeneration(config=self.model_config)

        # Shared embedding setting
        self.embeddings = self.model.model.shared
        # Encoder Setting
        self.encoder_model = self.model.get_encoder()
        # Dimension Setting
        self.d_hidden = self.encoder_model.embed_tokens.embedding_dim
        # Decoder Setting
        self.decoder_model = self.model.get_decoder()
        # Final Layer Setting
        self.lm_head = self.model.lm_head

        # Variational model setting
        self.variational = variational
        if self.variational:
            self.context_to_mu = nn.Linear(self.d_hidden, d_latent)
            self.context_to_logvar = nn.Linear(self.d_hidden, d_latent)
            self.mu_to_context = nn.Linear(d_latent, self.d_hidden)
            self.logvar_to_context = nn.Linear(d_latent, self.d_hidden)

            self.kl_criterion = GaussianKLLoss()

    def forward(self, src_input_ids, src_attention_mask, trg_input_ids, trg_attention_mask, 
                non_pad_position=None, tgt_subsqeunt_mask=None):

        # Pre_setting for variational model and translation task
        trg_input_ids_copy = torch.clone(trg_input_ids)
        trg_attention_mask_copy = torch.clone(trg_attention_mask)
        trg_input_ids = trg_input_ids[:, :-1]
        trg_attention_mask = trg_attention_mask[:, :-1]

        # Input and output embedding sharing mode
        if self.emb_src_trg_weight_sharing:
            src_input_embeds = self.embeddings(src_input_ids)
            trg_input_embeds = self.embeddings(trg_input_ids)
            trg_input_embeds_ = self.embeddings(trg_input_ids_copy)

        # Encoder Forward
        if self.emb_src_trg_weight_sharing:
            src_encoder_out = self.encoder_model(inputs_embeds=src_input_embeds,
                                                 attention_mask=src_attention_mask)
            src_encoder_out = src_encoder_out['last_hidden_state']
        else:
            src_encoder_out = self.encoder_model(input_ids=src_input_ids,
                                                 attention_mask=src_attention_mask)
            src_encoder_out = src_encoder_out['last_hidden_state']

        if self.variational:
            # Source sentence latent mapping
            src_mu = self.context_to_mu(src_encoder_out)
            src_logvar = self.context_to_logvar(src_encoder_out)
            # Target sentence latent mapping
            with torch.no_grad():
                if self.emb_src_trg_weight_sharing:
                    trg_encoder_out = self.encoder_model(inputs_embeds=trg_input_embeds_,
                                                        attention_mask=trg_attention_mask_copy)
                    trg_encoder_out = trg_encoder_out['last_hidden_state']
                else:
                    trg_encoder_out = self.encoder_model(input_ids=trg_input_ids_copy,
                                                         attention_mask=trg_attention_mask_copy)
                    trg_encoder_out = trg_encoder_out['last_hidden_state']

            trg_mu = self.context_to_mu(trg_encoder_out)
            trg_logvar = self.context_to_logvar(trg_encoder_out)

            kl = self.kl_criterion(src_mu, src_logvar, trg_mu, trg_logvar)

            mu = self.mu_to_context(src_mu)
            logvar = self.logvar_to_context(src_logvar)

            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(mu)

            src_encoder_out = torch.add(src_encoder_out, z)
        else:
            kl = 0

        # Decoder
        if self.emb_src_trg_weight_sharing:
            model_out = self.decoder_model(inputs_embeds=trg_input_embeds, 
                                           attention_mask=trg_attention_mask,
                                           encoder_hidden_states=src_encoder_out,
                                           encoder_attention_mask=src_attention_mask)
            model_out = self.lm_head(model_out['last_hidden_state'])
        else:
            model_out = self.decoder_model(input_ids=trg_input_ids, 
                                           attention_mask=trg_attention_mask,
                                           encoder_hidden_states=src_encoder_out,
                                           encoder_attention_mask=src_attention_mask)
            model_out = self.lm_head(model_out['last_hidden_state'])

        if non_pad_position is not None:
            model_out = model_out[non_pad_position]

        return model_out, kl