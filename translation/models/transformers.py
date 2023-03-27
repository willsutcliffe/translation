import torch.nn as nn

class Transformer(nn.Module):

    def __init__(self, encoder, decoder):
        """
        Constructor for the Transformer
        :param encoder: Encoder
        :param decoder: Decoder
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_input, dec_input, enc_mask, dec_mask):
        """

        :param enc_input: torch.Tensor
        :param dec_input: torch.Tensor
        :param enc_mask: torch.Tensor
        :param dec_mask: torch.Tensor
        :return: torch.Tensor
        """
        enc_output = self.encoder(enc_input, enc_mask)
        dec_output = self.decoder(enc_output, dec_input, enc_mask, dec_mask)
        return dec_output


