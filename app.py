import streamlit as st
import numpy as np
import torch
from transformers import pipeline
from translation.models.encoder import Encoder
from translation.models.decoder import Decoder
from translation.models.transformers import Transformer
from transformers import AutoTokenizer

model_checkpoint = f"Helsinki-NLP/opus-mt-en-ru"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

device = "cpu"
encoder = Encoder(vocab_size=tokenizer.vocab_size + 1,
                  max_len=512,
                  d_k=16,
                  d_model=64,
                  n_heads=8,
                  n_layers=4,
                  dropout_prob=0.1)
decoder = Decoder(vocab_size=tokenizer.vocab_size + 1,
                  max_len=512,
                  d_k=16,
                  d_model=64,
                  n_heads=8,
                  n_layers=4,
                  dropout_prob=0.1)
transformer = Transformer(encoder, decoder)
transformer.load_state_dict(torch.load('en_ru_transformer_8epochs.pt'))
transformer.eval()
transformer.to(device)

def translate(input_sentence):
  # get encoder output first
    enc_input = tokenizer(input_sentence, return_tensors='pt').to(device)
    #print(device)
    enc_output = encoder(enc_input['input_ids'], enc_input['attention_mask'])

    # setup initial decoder input
    dec_input_ids = torch.tensor([[ int(tokenizer.vocab_size)]], device=device)
    dec_attn_mask = torch.ones_like(dec_input_ids, device=device)

  # now do the decoder loop
    for _ in range(32):
        dec_output = decoder(
            enc_output,
            dec_input_ids,
            enc_input['attention_mask'],
            dec_attn_mask,
        )

        # choose the best value (or sample)
        prediction_id = torch.argmax(dec_output[:, -1, :], axis=-1)

        # append to decoder input
        dec_input_ids = torch.hstack((dec_input_ids, prediction_id.view(1, 1)))

        # recreate mask
        dec_attn_mask = torch.ones_like(dec_input_ids)

        # exit when reach </s>
        if prediction_id == 0:
            break
  
    translation = tokenizer.decode(dec_input_ids[0, 1:])
    #print(translation)
    return(translation)

st.title("Language-Translator")

# setting up the dropdown list of the languages

option = st.selectbox(
    'Which language would you choose to type',
    ('English', 'Russian'))

option1 = st.selectbox('Which language would you like to translate to',
                       ('English','Russian'))


sent = "Enter the text in "+option+" language in the text-area provided below"

# setting up the dictionary of languages to their keywords



sentence = st.text_area(sent, height=250)

if st.button("Translate"):

    try:

        if option == option1:
            st.write("Please Select different Language for Translation")

        else:
            ans = translate(sentence)
            st.write(ans)
            

    except:
        st.write("Please do cross check if text-area is filled with sentences or not")
