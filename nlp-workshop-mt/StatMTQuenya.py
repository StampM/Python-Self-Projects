#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 20:03:34 2020

@author: MStamp
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 11:28:17 2020

@author: MStamp
"""

# = Libraries to read in
import os, sys

# === key keras parts to create model ===

from keras.models import Model
from keras.layers import Input, LSTM, GRU, Dense, Embedding
from keras.preprocessing.text import Tokenizer # use this instead of nltk
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# put in matrix format and plot networks
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# read in data

quenya_dict_source = 'Quenya_dict.csv'

quenya_dict = pd.read_csv(quenya_dict_source, dtype = 'str')


# apply input and output sentences
inputs = list(quenya_dict['english'])
outputs = [str(word) + ' <eos>' for word in quenya_dict['quenya']]
output_sent_inputs = ['<sos> ' + str(word) for word in quenya_dict['quenya']]

ex_val = 546
print(inputs[ex_val] + '\n' + outputs[ex_val] + '\n' + output_sent_inputs[ex_val])

# ==== Tokenization & Padding 

Total_inputs = len(inputs)

input_tokenizer = Tokenizer(num_words = Total_inputs)

# have to change code type
inputs = [str(word) for word in inputs]

input_tokenizer.fit_on_texts(inputs) # apply with inputs in English - issue as float object has no attribute

input_int_seq = input_tokenizer.texts_to_sequences(inputs)
input_int_seq[ex_val]


word2idx_inputs = input_tokenizer.word_index
# - word2idx_inputs[ex_val]

word2idx_inputs[inputs[ex_val].lower().split(' ')[0]]

# == apply same to output

Total_outputs = len(outputs)

output_tokenizer = Tokenizer(num_words=Total_outputs, filters='') #
output_tokenizer.fit_on_texts(outputs + output_sent_inputs)
# should be here as combi


output_int_seq = output_tokenizer.texts_to_sequences(outputs)
output_input_int_seq = output_tokenizer.texts_to_sequences(output_sent_inputs)

word2idx_outputs = output_tokenizer.word_index

# === Padding need to get max lens in order to correctly pad sequences
max_input_len = max(len(sen) for sen in input_int_seq)
max_output_len = max(len(sen) for sen in output_int_seq) # not output - input - would expect to be longer as have <eos> at end
# smaller in this case - as just word for word changes


encoder_input_seq = pad_sequences(input_int_seq, maxlen = max_input_len)
encoder_input_seq.shape
encoder_input_seq[ex_val]
# as only two words - see three output
decoder_input_seq = pad_sequences(output_input_int_seq, maxlen = max_output_len, padding = 'post')
decoder_input_seq[ex_val] # only two seen in any output


# === much larger output values - can see longer sentence structure

# from numpy import array
from numpy import asarray
from numpy import zeros

embeddings_dictionary = dict()

glove_source = '/Users/MStamp/Documents/AIA Workshop/NLP Local Programme Setup/Code Structure/Pre Made Translation Corpus/glove.42B.300d.txt'

# open
glove_file =  open(glove_source, encoding="utf8")

# get dim output
for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary[word] = vector_dimensions
glove_file.close()

# == 

num_words = min(Total_inputs, len(word2idx_inputs) + 1)

emb_size = 300

embedding_matrix = zeros((num_words, emb_size))


for word,index in word2idx_inputs.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None: # is there some instant of no words having an output
        embedding_matrix[index] = embedding_vector # vector does not match output

example_text = 'fridge'

print(embeddings_dictionary[example_text])

# ===== Statistical Based MT Instead =====

# rather than applying Neural based - can simpler MT with statistical NLP models be applied


# easiest example given - 
def simple_model(input_shape, output_sequence_length, english_vocab_size, french_vocab_size):
    learning_rate = 1e-3
    input_seq = Input(input_shape[1:])
    rnn = GRU(64, return_sequences = True)(input_seq) # simple start - gated recurrent 
    logits = TimeDistributed(Dense(french_vocab_size))(rnn)
    model = Model(input_seq, Activation('softmax')(logits))
    model.compile(loss = sparse_categorical_crossentropy, 
                 optimizer = Adam(learning_rate), 
                 metrics = ['accuracy'])
    
    return model
# tests.test_simple_model(simple_model)
tmp_x = pad(preproc_navilist, max_navi_sequence_length)
tmp_x = tmp_x.reshape((-1, preproc_navilist.shape[-2], 1))
# Train the neural network
simple_rnn_model = simple_model(
    tmp_x.shape,
    max_navi_sequence_length,
    english_vocab_size,
    navi_vocab_size)
simple_rnn_model.fit(tmp_x, preproc_navilist, batch_size=1024, epochs=10, validation_split=0.2)
# Print prediction(s)
print(logits_to_text(simple_rnn_model.predict(tmp_x[:1])[0], french_tokenizer))


# == stat models