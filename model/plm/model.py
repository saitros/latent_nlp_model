import math
# Import PyTorch
import torch
import torch.nn as nn
# Import Huggingface
# T5
from transformers import T5ForConditionalGeneration, T5EncoderModel, T5Config, T5TokenizerFast
# Bart
from transformers import BartTokenizerFast, BartForConditionalGeneration, BartConfig

class Pretrained_Transformer(nn.Module):
    def __init__(self, model_type, decoder_type, isPreTrain, d_latent, device):
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
        self.model_type = model_type
        self.isPreTrain = isPreTrain
        self.device = device
        self.decoder_type = decoder_type

        if self.model_type == 'T5':
            if self.isPreTrain:
                self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
            else:
                model_config = T5Config("t5-small")
                self.model = T5ForConditionalGeneration(config=model_config)

            # Encoder1 Setting
            self.encoder1_embedding = self.model1.encoder.embed_tokens
            self.encoder1_model = self.model1.encoder.block
            self.encoder1_final_layer_norm = self.model1.encoder.final_layer_norm
            self.encoder1_dropout = self.model1.encoder.dropout

        elif self.model_type == 'bart':
            if self.isPreTrain:
                self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
            else:
                model_config = BartConfig('facebook/bart-base')
                self.model = BartForConditionalGeneration(config=model_config)


    def forward(self, input_ids, attention_mask):

    #===================================#
    #=============== T5 ================#
    #===================================#

        if self.model_type == 'T5':
            # Encoder1 Forward
            wae_enc_out = self.encoder1_embedding(input_ids)
            new_attention_mask = self.model1.get_extended_attention_mask(attention_mask, 
                                                                         attention_mask.shape, self.device)
            for i in range(len(self.encoder1_model)):
                wae_enc_out, _ = self.encoder1_model[i](hidden_states=wae_enc_out, 
                                                        attention_mask=new_attention_mask)

            wae_enc_out = self.encoder1_final_layer_norm(wae_enc_out)
            wae_enc_out = self.encoder1_dropout(wae_enc_out)

            # Encoder2 Forward
            wae_dec_out = self.encoder2_model(inputs_embeds=wae_enc_out, 
                                              attention_mask=attention_mask)
            wae_dec_out = wae_dec_out['last_hidden_state']

            # Decoder
            if self.decoder_type == 'Transformer':
                model_out = self.decoder_model(inputs_embeds=wae_enc_out, 
                                            attention_mask=attention_mask,
                                            encoder_hidden_states=wae_dec_out,
                                            encoder_attention_mask=attention_mask)
                model_out = self.lm_head(model_out['last_hidden_state'])
            else:
                model_out, _ = self.decoder_model(wae_enc_out)
                model_out = self.decoder_linear1(model_out)
                model_out = self.decoder_linear2(model_out)

            return wae_enc_out, wae_dec_out, model_out