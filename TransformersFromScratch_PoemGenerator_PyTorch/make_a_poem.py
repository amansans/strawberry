# -*- coding: utf-8 -*-
"""Make_A_Poem.ipynb

Automatically generated by Colaboratory.

### **Upload Libraries**
"""

import torch
import torch.nn as nn
import os
import time
import pandas as pd
import gc
import tensorflow as tf
from tensorflow.keras import backend as K
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import math
import optuna
import numpy as np
import torch.nn.functional as F
import re
from torch.utils.data import DataLoader

!pip install torch==1.4.0

!pip install optuna

!nvidia-smi --query-gpu=gpu_name,driver_version,memory.total --format=csv

"""### **Upload Poem Data**"""

NUM_SAMPLES = 100
MAX_VOCAB_SIZE = 20000

BATCH_SIZE = 10
EPOCHS = 10

file_name = '/content/drive/MyDrive/PoetryFoundationData.csv'

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(file_name,nrows=NUM_SAMPLES,encoding='utf-8')
df = df.sample(frac=1)
df.head()

def preprocess_sentence(sentence):
    
    sentence = re.sub(r"'", "", sentence)
    sentence = re.sub(r"([?.!,¿])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,¿]+", " ", sentence)
    sentence = sentence.strip()
    
    return sentence

input_data = []
target_data = []
target_input_data = []

for poem_number in range(len(df['Poem'])):
    count = 0
    for line in df['Poem'][poem_number].split('.'):
        
        if len(preprocess_sentence(line)) > 650:
            break
        if count == 0:
            count = count + 1
            previous_line = preprocess_sentence(line)
            continue
        
        new_line = preprocess_sentence(line)
        
        if new_line == " ":
            continue
            
        #adding <sos> and <eos> to the decoder inputs
        new_line_source = new_line + ' <eos>' 
        new_line_target = '<sos> ' + new_line
        
        #adding preprocessed data to a list
        input_data.append(previous_line)
        target_data.append(new_line_source)
        target_input_data.append(new_line_target)
        
        previous_line = new_line
        count = count + 1                

del df
gc.collect()

input_data = np.array(input_data)
target_data = np.array(target_data)
target_input_data = np.array(target_input_data)

#Input tokens
tokenizer_inputs = Tokenizer(num_words=MAX_VOCAB_SIZE, filters='')
tokenizer_inputs.fit_on_texts(input_data)

input_sequences = tokenizer_inputs.texts_to_sequences(input_data)
input_max_len = max(len(s) for s in input_sequences)

#Output tokens
tokenizer_outputs = Tokenizer(num_words= MAX_VOCAB_SIZE, filters='')
tokenizer_outputs.fit_on_texts(target_data)
tokenizer_outputs.fit_on_texts(target_input_data)

target_sequences = tokenizer_outputs.texts_to_sequences(target_data)
target_sequences_inputs = tokenizer_outputs.texts_to_sequences(target_input_data)
target_max_len = max(len(s) for s in target_sequences)

print('Max Input Length: ', input_max_len)
print('Max Target Length: ', target_max_len)
print(target_data[10])
print(target_sequences[10])
print(target_input_data[10])
print(target_sequences_inputs[10])

word2idx_inputs = tokenizer_inputs.word_index
print('Found {} unique input tokens.'.format(len(word2idx_inputs)))
word2idx_outputs = tokenizer_outputs.word_index
print('Found {} unique output tokens.'.format(len(word2idx_outputs)))

num_words_inputs = len(word2idx_inputs) + 1
num_words_output = len(word2idx_outputs) + 1

encoder_inputs = torch.from_numpy(pad_sequences(input_sequences,maxlen=input_max_len,padding='post')).to(torch.int64).to(device)
decoder_inputs = torch.from_numpy(pad_sequences(target_sequences_inputs,maxlen=target_max_len,padding='post')).to(torch.int64).to(device)
decoder_targets = torch.from_numpy(pad_sequences(target_sequences, maxlen=target_max_len, padding='post')).to(torch.int64).to(device)

dataset = [None] * encoder_inputs.shape[0]
for i in range(encoder_inputs.shape[0]):
    dataset[i] = [encoder_inputs[i], decoder_inputs[i], decoder_targets[i]]

"""
## **Create Embedding Matrix using pretrained Glove Embeddings**

"""

embeddings_index = {}
glove_file = '/content/drive/MyDrive/glove.6B.300d.txt' 
file = open((glove_file),encoding="utf8")

for line in file:
    values = line.split()
    word = values[0]
    coeff = np.asarray(values[1:257], dtype = 'float32')
    coeff = torch.from_numpy(coeff)
    embeddings_index[word] =  coeff
file.close()

embedding_matrix_inp = torch.zeros(len(tokenizer_inputs.index_word) + 1,256)

for i in range (1,len(tokenizer_inputs.index_word) + 1):
    embedding_vector = embeddings_index.get(tokenizer_inputs.index_word.get(i))
    if embedding_vector is not None:
        try:
          embedding_matrix_inp[i]  =  embedding_vector
        except:
          embedding_matrix_inp[i] = torch.zeros(256)+ (torch.rand(256))/100

embedding_matrix_out = torch.zeros(len(tokenizer_outputs.index_word) + 1,256)

for i in range (1,len(tokenizer_outputs.index_word) + 1):
    embedding_vector = embeddings_index.get(tokenizer_outputs.index_word.get(i))
    if embedding_vector is not None:
        try:
          embedding_matrix_out[i]  =  embedding_vector
        except:
          embedding_matrix_out[i] = torch.zeros(256)+ (torch.rand(256))/100

embedding_matrix_inp.to(device)
embedding_matrix_out.to(device)

"""# **Create model**"""

class PositionalEncoder(nn.Module):
    def __init__(self, embedding_size, max_len):
        super().__init__()
        self.embedding_size = embedding_size
        self.max_len = max_len
        
        self.pe = torch.zeros(self.max_len,self.embedding_size)
        for pos in range(self.max_len):
            for i in range(0,self.embedding_size,2):
                self.pe[pos,i] = math.sin(pos/(10000 ** ((2*i)/self.embedding_size)))
                self.pe[pos,i + 1] = math.cos(pos/(10000 ** ((2*(i+1))/self.embedding_size)))
        self.pe = self.pe[None, :].to(device)
        
    def forward(self,x):
        seq_length =x.shape[1]
        x = x + self.pe[:,:seq_length]
        return x

class SelfAttention(nn.Module):
    def __init__(self, embedding_size, number_heads):
        super(SelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.number_heads = number_heads
        self.head_dim = embedding_size // number_heads

        assert (self.head_dim * number_heads == embedding_size), "Embedding size indivisible by number of heads"

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.FC = nn.Linear(number_heads * self.head_dim, embedding_size)

    def forward(self, values, keys, query, mask):
        #Number of training examples
        N = query.shape[0]

        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        values = values.reshape(N, value_len, self.number_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.number_heads, self.head_dim)
        query = query.reshape(N, query_len, self.number_heads, self.head_dim)
        
        #Passing them through the Linear Weights --> eg. K = k X W(k)
        values = self.values(values)  
        keys = self.keys(keys) 
        queries = self.queries(query) 

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))


        attention = torch.softmax(energy / (self.embedding_size ** (1 / 2)), dim=3)
        
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values]).reshape(N, query_len, self.number_heads * self.head_dim)

        out = self.FC(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, number_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embedding_size, number_heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion * embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_size, embedding_size),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        
        out = self.dropout(self.norm2(forward + x))
        return out

class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embedding_size,
        num_layers,
        number_heads,
        device,
        forward_expansion,
        dropout,
        max_length,
    ):

        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.word_embedding = nn.Embedding.from_pretrained(embedding_matrix_inp, padding_idx=0)
        self.position_embedding = PositionalEncoder(embedding_size, max_length)
        
        #init weights for multiple layers
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embedding_size,number_heads,dropout=dropout,forward_expansion=forward_expansion,) 
                for _ in range(num_layers)
            ]
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        x = self.word_embedding(x).to(device)
        x = self.position_embedding(x).to(device)
        out = self.dropout(x)

        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out

class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, number_heads, forward_expansion, dropout, device):
        super(DecoderBlock, self).__init__()
        self.norm = nn.LayerNorm(embedding_size)
        self.attention = SelfAttention(embedding_size, number_heads=number_heads)
        self.transformer_block = TransformerBlock(
            embedding_size, number_heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, tar_mask):
        attention = self.attention(x, x, x, tar_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out

class Decoder(nn.Module):
    def __init__(
        self,
        tar_vocab_size,
        embedding_size,
        num_layers,
        number_heads,
        forward_expansion,
        dropout,
        device,
        max_length,
    ):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding.from_pretrained(embedding_matrix_out, padding_idx=0)
        self.position_embedding = PositionalEncoder(embedding_size, max_length)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embedding_size, number_heads, forward_expansion, dropout, device)
                for _ in range(num_layers)
            ]
        )
        self.FC = nn.Linear(embedding_size, tar_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tar_mask):
        N, seq_length = x.shape
        x = self.word_embedding(x)
        x = self.position_embedding(x).to(device)
        x = self.dropout(x).to(device)
        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, tar_mask)

        out = self.FC(x)

        return out

class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tar_vocab_size,
        src_pad_idx,
        tar_pad_idx,
        embedding_size=256,
        num_layers=4,
        forward_expansion=4,
        number_heads=2,
        dropout=0,
        device="cpu",
        max_length=9,
    ):

        super(Transformer, self).__init__()

        self.encoder = Encoder(
            src_vocab_size,
            embedding_size,
            num_layers,
            number_heads,
            device,
            forward_expansion,
            dropout,
            max_length,
        )

        self.decoder = Decoder(
            tar_vocab_size,
            embedding_size,
            num_layers,
            number_heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.tar_pad_idx = tar_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_tar_mask(self, tar):
        N, tar_len = tar.shape
        tar_mask = torch.tril(torch.ones((tar_len, tar_len))).expand(N, 1, tar_len, tar_len)
        return tar_mask.to(self.device)

    def forward(self, src, tar):
        src_mask = self.make_src_mask(src)
        tar_mask = self.make_tar_mask(tar)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(tar, enc_src, src_mask, tar_mask)
        return out

"""### **Optimize model**"""

def train_valid_split(dataset, train_split = 0.8, valid_split = 0.2):
    
    train_set, valid_set = dataset[:int(len(dataset)*0.80)], dataset[(int(len(dataset)*0.80))+1:]
    train_set = DataLoader(train_set,batch_size=BATCH_SIZE, shuffle=False)
    valid_set = DataLoader(valid_set,batch_size=int(valid_split * len(dataset)), shuffle=False)
    
    return train_set, valid_set

def val_model(model,valid_set):

    tar_pad_idx=0
    model.eval()
    with torch.no_grad():
        
        for batch,(encoder_inputs,decoder_inputs,decoder_targets) in enumerate(valid_set):
            out = model(encoder_inputs,decoder_inputs[:,:-1]).to(device)
            target = decoder_targets[:,1:].contiguous().view(-1).to(device)
            loss = F.cross_entropy(out.view(-1, out.size(-1)),target, ignore_index=tar_pad_idx)
    
    return loss

def batch_train(optim,model,train_set,valid_set):
    
    i = 0
    total_loss = 0
    tar_pad_idx=0

    for batch,(encoder_inputs,decoder_inputs,decoder_targets) in enumerate(train_set):
        i = i + 1
        out = model(encoder_inputs,decoder_inputs[:,:-1]).to(device)
        target = decoder_targets[:,1:].contiguous().view(-1).to(device)
        optim.zero_grad()
        loss = F.cross_entropy(out.view(-1, out.size(-1)),target, ignore_index=tar_pad_idx)
        loss.backward()
        optim.step()
        
        total_loss =  total_loss + loss
        
    train_loss = total_loss/i
    total_loss = 0
    val_loss = val_model(model,valid_set)
    
    return train_loss, val_loss

def train_model(optim,model,epochs):
    
    model.train()
    start = time.time()
    temp = start
    train_set, valid_set = train_valid_split(dataset)

    for epoch in range(epochs):
        train_loss, val_loss = batch_train(optim,model,train_set,valid_set)
        print("Time_elspased '{ti}' \t Epoch: '{ep}' \t train_Loss: '{lo}' \t valid_Loss: '{vlo}'".format(ti = time.strftime("%H:%M:%S", time.gmtime(time.time() - temp)), ep = epoch + 1, lo = train_loss, vlo = val_loss))
        
    return train_loss, val_loss

def objective(trial):
    
    number_heads = trial.suggest_int("Num_heads",low=2,high=4,step=2)
    forward_expansion = trial.suggest_int("Forward_expansion",1,4)
    num_layers = trial.suggest_int("N_layers",1,6)
    dropout_rate = trial.suggest_float('Dropout_rate',0.0,0.95)
    learning_rate = trial.suggest_float("Adam_learning_rate", 1e-5, 1e-1, log=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = "cpu"
    model = Transformer(num_words_inputs,
                        num_words_output,
                        src_pad_idx=0,
                        tar_pad_idx=0,
                        num_layers=num_layers,
                        forward_expansion=forward_expansion,
                        number_heads=number_heads,
                        dropout= dropout_rate,
                        device=device,
                        max_length= target_max_len).to(device)
    
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    train_loss, val_loss = train_model(optim,model,250)
    
    del model
    gc.collect()
    K.clear_session()
    torch.cuda.empty_cache()
    
    return (abs(train_loss - val_loss)*20) + train_loss

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = "cpu"
model = Transformer(num_words_inputs,
                        num_words_output,
                        src_pad_idx=0,
                        tar_pad_idx=0,
                        device=device,
                        max_length= target_max_len).to(device)
    
optim = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98), eps=1e-9)
train_model(model,75)

#Need to do something about train and val data source
if __name__ == "__main__":
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("{} : {}".format(key, value))

"""### **Test Model**"""

def test(model, user_input, max_len=target_max_len, num_lines = 3):
    
    model.eval()
    user_input = list(str.split(user_input, "\n"))
    df = pd.DataFrame(user_input)
    cleaned_user_input = df[0].apply(lambda x : preprocess_sentence(x))
    tar = ['<sos>']
    
    del df
    gc.collect()

    cleaned_user_input = tokenizer_inputs.texts_to_sequences(cleaned_user_input)
    src = torch.Tensor(cleaned_user_input[0])
    src = torch.unsqueeze(src,0).long()
    
    outputs = torch.zeros(max_len)
    outputs = torch.unsqueeze(outputs,0).long()
    
    reversed_out_tok = dict(map(reversed, tokenizer_outputs.word_index.items()))
    reversed_out_tok
    
    EOS_index = tokenizer_outputs.word_index.get('<eos>')
    SOS_index = tokenizer_outputs.word_index.get('<sos>')
    
    outputs[0][0] = SOS_index

#add another loop for multiple line outputs

    for i in range(1, target_max_len):
        out = model(src,outputs[:i])
        out = F.softmax(out, dim=-1)
        val,m_id = torch.topk(out[:,-1],1)
        outputs[0][i] = m_id

        #Uncomment this once the mode is kinda trained
        # if (int(m_id) == EOS_index):
        #     break
        
    return ' '.join([reversed_out_tok.get(int(x)) for x in torch.squeeze(outputs)[1:i]])

test(model, user_input="It was the season of love")