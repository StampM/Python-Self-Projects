#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 21:16:15 2020

@author: mstamp
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

dothraki_dict_source = 'dothraki_dict.csv'

# will have issue with text after if to not specify string type
os.chdir('/Users/MStamp/Documents/AIA Workshop/NLP Local Programme Setup/Code Structure/Attempts of Translation')

dothraki_dict = pd.read_csv(dothraki_dict_source, dtype = 'str')

# == alt - have .txt file as input

# apply input and output sentences
inputs = list(dothraki_dict['english'])
outputs = [str(word) + ' <eos>' for word in dothraki_dict['dothraki']]
output_sent_inputs = ['<sos> ' + str(word) for word in dothraki_dict['dothraki']]


print(str(len(inputs)) + ' ' + str(len(outputs)) + ' ' + str(len(output_sent_inputs)))

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


# ==== Apply Word Embeddings ====


# == get glove file
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

embedding_layer = Embedding(num_words, emb_size , weights=[embedding_matrix], input_length=max_input_len)


# ==== develop model - additional structure featurea ====

num_words_output = len(word2idx_outputs) + 1

decoder_targets_one_hot = zeros((
        len(inputs),
        max_output_len,
        num_words_output
    ),
    dtype='float32'
)
# to fill with training output

for i, d in enumerate(decoder_input_seq):
    for t, word in enumerate(d):
        decoder_targets_one_hot[i, t, word] = 1
        
decoder_targets_one_hot[ex_val] 

# === encoder input 

nodes = 256 # unsure of impact to change this

encoder_inputs_placeholder = Input(shape=(max_input_len,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = LSTM(nodes, return_state=True)

encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]

# ==== define decoder

decoder_inputs_placeholder = Input(shape=(max_output_len,))

decoder_embedding = Embedding(num_words_output, nodes)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

decoder_lstm = LSTM(nodes, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)


# ==== Combine for Prediction and Error 

decoder_dense = Dense(num_words_output, activation = 'softmax')

decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs_placeholder, 
  decoder_inputs_placeholder], decoder_outputs)
model.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

from keras.utils import plot_model
plot_model(model, to_file='model_plot_dothraki.png', show_shapes=True, show_layer_names=True)

# = Fit Model

batch_size = 64
epochs = 1

r = model.fit(
    [encoder_input_seq, decoder_input_seq],
    decoder_targets_one_hot,
    batch_size=batch_size,
    epochs=epochs,
    validation_split=0.1,
)


# Comment - graph couldn't be sorted in topological order

# Warning - Converting sparse IndexedSlices to a dense Tensor of unknown shape. This may consume a large amount of memory.

# also reconnected the python concole and lost all variables

# ==== Model Prediction Re-format

encoder_model = Model(encoder_inputs_placeholder, encoder_states) # apply fitted output now to states results

# Since now at each step we need the decoder hidden and cell states, we will modify our model to accept the hidden and cell states as shown below:

decoder_state_input_h = Input(shape=(nodes,))
decoder_state_input_c = Input(shape=(nodes,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

# Now at each time step, there will be only single word in the decoder input, we need to modify the decoder embedding layer as follows:

decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

# Next, we need to create the placeholder for decoder outputs:

decoder_outputs, h, c = decoder_lstm(decoder_inputs_single_x, initial_state=decoder_states_inputs)

# To make predictions, the decoder output is passed through the dense layer:

decoder_states = [h, c]
decoder_outputs = decoder_dense(decoder_outputs)

# The final step is to define the updated decoder model, as shown here:

decoder_model = Model(
    [decoder_inputs_single] + decoder_states_inputs,
    [decoder_outputs] + decoder_states
)

# decoder model is what we will use in displaying output now

# === Make Predictions =====

idx2word_input = {v:k for k, v in word2idx_inputs.items()}
idx2word_target = {v:k for k, v in word2idx_outputs.items()}

# have user input given
def translate_own_sentence():
    print('Put your own sentence')
    input_seq = input()
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []

    for _ in range(max_output_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''

        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)

translate_own_sentence()