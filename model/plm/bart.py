import math
# Import PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
# Import Huggingface
from transformers import BartForConditionalGeneration, BartConfig
#
from ..custom_transformer.latent_module import Latent_module

class custom_Bart(nn.Module):
    def __init__(self, isPreTrain, variational_mode, d_latent, emb_src_trg_weight_sharing=True):
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
        self.model_config.use_cache = False

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
        self.variational_mode = variational_mode
        self.latent_module = Latent_module(self.d_hidden, d_latent, variational_mode)

    def forward(self, src_input_ids, src_attention_mask, trg_input_ids, trg_attention_mask, 
                non_pad_position=None, tgt_subsqeunt_mask=None):

        # Pre_setting for variational model and translation task
        trg_input_ids_copy = torch.tensor(trg_input_ids, requires_grad=True)
        trg_attention_mask_copy = torch.tensor(trg_attention_mask, requires_grad=True)
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

            # Variational
            if self.variational_mode != 0:
                # Target sentence latent mapping
                with torch.no_grad():
                    if self.emb_src_trg_weight_sharing:
                        encoder_out_trg = self.encoder_model(inputs_embeds=trg_input_embeds_,
                                                             attention_mask=trg_attention_mask_copy)
                        encoder_out_trg = src_encoder_out['last_hidden_state']
                    else:
                        encoder_out_trg = self.encoder_model(input_ids=trg_input_ids_copy,
                                                             attention_mask=trg_attention_mask_copy)
                        encoder_out_trg = src_encoder_out['last_hidden_state']

                encoder_out, dist_loss = self.latent_module(encoder_out, encoder_out_trg)
            else:
                dist_loss = torch.tensor(0, dtype=torch.float)

        # Decoder
        if self.emb_src_trg_weight_sharing:
            model_out = self.decoder_model(inputs_embeds=trg_input_embeds, 
                                           attention_mask =trg_attention_mask,
                                           encoder_hidden_states=src_encoder_out,
                                           encoder_attention_mask=src_attention_mask)
            model_out = self.lm_head(model_out['last_hidden_state'])
        else:
            model_out = self.decoder_model(input_ids=trg_input_ids, 
                                           attention_mask =trg_attention_mask,
                                           encoder_hidden_states=src_encoder_out,
                                           encoder_attention_mask=src_attention_mask)
            model_out = self.lm_head(model_out['last_hidden_state'])

        if non_pad_position is not None:
            model_out = model_out[non_pad_position]

        return model_out, dist_loss

    @staticmethod
    def generate_square_subsequent_mask(sz, device):
        mask = torch.tril(torch.ones(sz, sz, dtype=torch.float, device=device))
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, 0.0)
        return mask