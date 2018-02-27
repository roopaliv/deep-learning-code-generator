#!/usr/bin/python
# coding: utf-8
from __future__ import print_function, division
import numpy
import os
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM, Flatten
from keras.callbacks import ModelCheckpoint
from keras.layers import Activation
import string
from utils.text import get_vocab, get_inverse_vocab
import sys
import argparse
import re
from utils.multi_gpu import make_parallel
import matplotlib.pyplot as plt

inverse_vocab = {}
vocab = {}
len_vocab = len(vocab)
vocab_len = 0
#print(inverse_vocab)
def get_raw_data(data_dir, data_ext):
	# Preprocessing the raw data
	files = []
	files += [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.'+data_ext)]
	data = []
	for file_path in files:
		data.extend(open(file_path, 'r').read())
		print('######### End of Reading file: ' + file_path + ' ########')

	return data;



def get_training_data(data, len_sequence):
	X = []
	Y = []
	global inverse_vocab, vocab_len, vocab
	unq_chars = sorted(list(set(data)))
	vocab_len = len(unq_chars)
	vocab = get_vocab(unq_chars)
	inverse_vocab = get_inverse_vocab(unq_chars)

	for i in range(0, len(data)-len_sequence):
		X.append([vocab[value] for value in data[i:i + len_sequence]])
		Y.append([vocab[value] for value in data[i + len_sequence]])
	### reshape to (number of sequences, length of sequence, number of features) then normalize
	x = numpy.reshape(X, (len(X), len_sequence, 1)) #/ float(len_vocab)
	x = x / float(vocab_len)

	# Converting the output to oneHot Encoding vector
	y = np_utils.to_categorical(Y, num_classes = vocab_len) # - onehot-
	return X, Y, x, y

def create_network(x,y, network_specs):

	#Using sequential model
	model = Sequential()

	### input shape (length of sequence, number of features)
	# Add a LSTM network
	model.add(LSTM(network_specs['dimension_of_lstm_layer'], input_shape=(x.shape[1], x.shape[2]), return_sequences=True))

	# Add a dropout for regularization
	model.add(Dropout(network_specs['dropout_for_hidden_layers']))

	# Using relu as an activation function for non-linearity
	model.add(Activation('relu'))

	# Adding rest of the configured LSTM layers
	for i in range(network_specs['no_of_lstm_layers'] - 1):
		model.add(LSTM(network_specs['dimension_of_lstm_layer'], return_sequences=True))
		model.add(Dropout(network_specs['dropout_for_hidden_layers']))
		model.add(Activation('relu'))

	# Flatten the output
	model.add(Flatten())

	# Add a Dense layer with softmax for output
	model.add(Dense(y.shape[1], activation='softmax'))
	return model

def compile_model(model):
	model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(lr=0.0001))





def test_model(model, X, cers):

	# Need to select the checkpoint file based on best parameters
	# Picking a random line as input from the text in order to predict the output
	#pattern = X[numpy.random.randint(0, len(X)-1)]
	pattern = X[0]

	# Mapping the input embedding to corresponding string
	global inverse_vocab, vocab_len
	input_str = [inverse_vocab[value] for value in pattern]

	result = []

	# To continue the predicton in continuation to the input string
	result.append(''.join(input_str))
	error_count = 0;
	prediction_extent = 200
	for i in range(prediction_extent):
		### reshape to (number of sequences =1, length of sequence=seq_len, number of features=1) then normalize
		x = numpy.reshape(pattern, (1, len(pattern), 1))
		x = x / float(vocab_len)
		x = numpy.array(x, dtype=numpy.float32)
		prediction = None
		# Inference on the defined model
		
		prediction = model.predict(x, verbose=0)

		# Output corresponds to the index of the OneHot encoding vector
		index = numpy.argmax(prediction)
		actual = X[i+1][-1]
		if(index!=actual):
			error_count+=1;
		# Collecting the predicted result
		result.append(inverse_vocab[index])

		# Updating the input string along with
		pattern.append(index)
		pattern = pattern[1:len(pattern)]
	char_error_rate = error_count/prediction_extent
	cers.append(char_error_rate)
	print("Character Error Rate: " + str(char_error_rate))
	print("Input: ", ''.join(input_str))
	print("Output: ", ''.join(result))
	return cers

if __name__ == "__main__":


	# Configuration settings for the Deep Neural Network model
	network_specs = {}
	network_specs['no_of_lstm_layers'] = 3
	network_specs['dimension_of_lstm_layer'] = 700
	network_specs['dropout_for_hidden_layers'] = 0.15
	network_specs['dimension_for_output'] = 1024

	parser = argparse.ArgumentParser(description='Usage of Deep Neural Model')
	parser.add_argument('--data_dir', default='data', type=str, help='path to dataset')
	parser.add_argument('--checkpoint_dir', type=str, help='checkpoint path')
	parser.add_argument('--seq_length', default=100, type=int, help='sequence length to train at a time')
	parser.add_argument('--data_ext', default='cpp', type=str, help='path to dataset')
	parser.add_argument('--batch_size', default=1, type=int, help='batch size')
	parser.add_argument('--num_gpus', default=1, type=int, help='num of gpus')


	args = parser.parse_args()



	files = []
	files += [os.path.join(args.checkpoint_dir, f) for f in os.listdir(args.checkpoint_dir) if f.endswith('.hdf5')]
	epochs = [];
	cers = [];
	for checkpoint_file in files:
		data = get_raw_data(args.data_dir, args.data_ext)
		X, Y, x, y = get_training_data(data, args.seq_length)
		X = X[:len(X)-len(X)%args.num_gpus]
		model = create_network(x,y, network_specs)
		epoch = int(re.findall(r'\d+', checkpoint_file)[0])
		print("#############################################################################")
		print("############### Results for Epoch "+str(epoch)+" start ######################")
		print("#############################################################################")
		if args.num_gpus != 1:
			model = make_parallel(model, args.num_gpus)
		batch_size = args.batch_size * args.num_gpus
		compile_model(model)
		model.load_weights(checkpoint_file)
		cers = test_model(model, X, cers)
		epochs.append(epoch)
		print("#############################################################################")
		print("############### Results for Epoch "+str(epoch)+" end #######################")
		print("#############################################################################")
	plt.xlabel('Epoch')
	plt.ylabel('Character Error rate')
	if args.data_dir[:-3]=='cpp':
		plt.title('Epoch vs Character Error Rate - Code')
	else:
		plt.title('Epoch vs Character Error Rate - Literature')
	xs, ys = zip(*sorted(zip(epochs, cers)))
	plt.plot(xs, ys, 'b-')
	#plt.show()
	plt.savefig(args.data_dir+'_EpochVsCer.png')





