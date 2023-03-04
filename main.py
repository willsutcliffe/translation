import torch
from translation.models.encoder import Encoder
from translation.models.decoder import Decoder
from translation.models.transformers import Transformer
from translation.training.littrainer import  LitTrainer
import numpy as np
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
import pytorch_lightning as pl

def preprocess_function(batch):
    model_inputs = tokenizer(
        batch['en'], max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(batch['ru'], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('PyCharm')





    raw_dataset = load_dataset('csv',
            data_files='test.csv')
    print(raw_dataset)
    lang_pair = "en-ru"
    model_checkpoint = f"Helsinki-NLP/opus-mt-{lang_pair}"
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    split = raw_dataset['train'].train_test_split(test_size=0.1, seed=42)
    max_input_length = 128
    max_target_length = 128

    tokenized_datasets = split.map(
        preprocess_function,
        batched=True,
        remove_columns=split["train"].column_names,
    )


    data_collator = DataCollatorForSeq2Seq(tokenizer)
    batch = data_collator([tokenized_datasets["train"][i] for i in range(0, 5)])
    batch.keys()


    train_loader = DataLoader(
        tokenized_datasets["train"],
        shuffle=True,
        batch_size=128,
        collate_fn=data_collator,
        num_workers=4, pin_memory=True,
    )
    valid_loader = DataLoader(
        tokenized_datasets["test"],
        batch_size=128,
        collate_fn=data_collator,
        num_workers=4, pin_memory=True
    )

    from translation.models.transformers import Transformer
    from translation.models.encoder import Encoder
    from translation.models.decoder import Decoder

    encoder = Encoder(vocab_size=tokenizer.vocab_size + 1,
                      max_len=512,
                      d_k=16,
                      d_model=64,
                      n_heads=4,
                      n_layers=2,
                      dropout_prob=0.1)
    decoder = Decoder(vocab_size=tokenizer.vocab_size + 1,
                      max_len=512,
                      d_k=16,
                      d_model=64,
                      n_heads=4,
                      n_layers=2,
                      dropout_prob=0.1)
    transformer = Transformer(encoder, decoder)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    encoder.to(device)
    decoder.to(device)
    model = LitTrainer(transformer, int(tokenizer.vocab_size + 1), tokenizer)
    trainer = pl.Trainer(
        precision=16,
        devices=1,
        accelerator="gpu",
        max_epochs=2
    )


    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader
                )



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
