import os
from torch import optim, nn, utils, Tensor
from torchvision.transforms import ToTensor
import pytorch_lightning as pl
import torch
from  torch.optim.lr_scheduler import ReduceLROnPlateau


class LitTrainer(pl.LightningModule):
    def __init__(self, model, vocab_size, tokenizer, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.criterion = criterion = nn.CrossEntropyLoss(ignore_index=-100)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        enc_input = batch['input_ids']
        enc_mask = batch['attention_mask']
        targets = batch['labels']
        dec_input = targets.clone().detach()
        dec_input = torch.roll(dec_input, shifts=1, dims=1)
        dec_input[:, 0] = self.vocab_size - 1
        dec_input = dec_input.masked_fill(
            dec_input == -100, self.tokenizer.pad_token_id)
        dec_mask = torch.ones_like(dec_input)
        dec_mask = dec_mask.masked_fill(dec_input == self.tokenizer.pad_token_id, 0)

        outputs = self.model(enc_input, dec_input, enc_mask, dec_mask)
        loss = self.criterion(outputs.transpose(2, 1), targets)

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        enc_input = batch['input_ids']
        enc_mask = batch['attention_mask']
        targets = batch['labels']
        dec_input = targets.clone().detach()
        dec_input = torch.roll(dec_input, shifts=1, dims=1)
        dec_input[:, 0] = self.vocab_size - 1
        dec_input = dec_input.masked_fill(
            dec_input == -100, self.tokenizer.pad_token_id)
        dec_mask = torch.ones_like(dec_input)
        dec_mask = dec_mask.masked_fill(dec_input == self.tokenizer.pad_token_id, 0)

        outputs = self.model(enc_input, dec_input, enc_mask, dec_mask)
        loss = self.criterion(outputs.transpose(2, 1), targets)

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer, 'min'),
                "monitor": "val_loss",
                "frequency": 1
            },
        }

