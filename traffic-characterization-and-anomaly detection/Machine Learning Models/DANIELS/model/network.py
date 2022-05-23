import tensorflow as tf
from keras.metrics import RootMeanSquaredError
from keras.models import Sequential
from keras.layers import Conv2D, LSTM, GRU, Dense, MaxPooling2D, Flatten, Dropout, Input
# from keras.losses import 

#from tcn import TCN

import os
from keras.backend import sigmoid
from keras.engine.input_layer import InputLayer

from keras.layers.core import Dropout
import _pickle

class Network:

    def __init__(self, kernel_size=[2],
                layers_type=['c', 'm', 'f', 'd', 'd', 'd'],
                layers_size=[16, 2, 0, 64, 16, 6],
                input_shape=(2,7,1),
                output_function=None,
                dilations=[1, 2, 4, 8, 16, 32],
                loss='mse',
                metrics=['mse'] , base_name='', dropout=0):

        self.network = Sequential()
        self.input_shape = input_shape
        self.name = base_name + 's_' + str(input_shape[0])
        self.metrics = metrics
        self.fitted = False
        self.dropout = dropout

        iter_layers = zip(layers_type, layers_size)
        if kernel_size != None: iter_kernel_size = iter(kernel_size)

        # Adicionando o input shape
        self.network.add(InputLayer(input_shape=self.input_shape))
        i = 0
        len_net = len(layers_type)
        # Adicionar camadas
        for layer_type, layer_size in iter_layers:
            i = i+1
            # Camadas Convolucionais
            if layer_type == 'c':
                self.network.add(Conv2D(filters=layer_size, kernel_size=next(iter_kernel_size), activation='relu', padding='same'))
            
            # Camada TCN
            elif layer_type == 't':
                self.network.add(TCN(nb_filters=layer_size, kernel_size=next(iter_kernel_size), activation='relu', padding='same', dilations=dilations))
            
            # Camada LSTM
            elif layer_type == 'l':
                self.network.add(LSTM(units=layer_size, activation='relu'))

            # Camada GRU
            elif layer_type == 'g':
                self.network.add(GRU(units=layer_size, activation='tanh', recurrent_activation=sigmoid))

            # Camada Densa
            elif layer_type == 'd':
                if i == len_net:
                    # Se for a última camada
                    
                    self.network.add(Dense(layer_size, activation=output_function))
                else:
                    if self.dropout > 0:
                        self.network.add(Dropout(self.dropout))
                    self.network.add(Dense(layer_size, activation='relu'))
            
            # Camada Max Pooling
            elif layer_type == 'm':
                self.network.add(MaxPooling2D(layer_size, padding='same'))
            
            # Camada Flatten
            elif layer_type == 'f':
                self.network.add(Flatten())
            
            else:
                print('Camada não definida no arquivo network.py')
                raise Exception('Erro network.py, camada: ' + layer_type)

            # Adicionar camada no nome do modelo
            self.name = self.name + '_' + layer_type + '_' + str(layer_size)
        
        # self.network.layers[-1].activation = None
        self.network.compile(loss=loss, optimizer='adam', metrics=self.metrics)
        #self.network.summary()

    @classmethod
    def from_file(cls, filename):
        with open(filename+'.pkl', 'rb') as _file:
            _self = _pickle.load(_file)

        #_self.network = tf.keras.models.load_model(filename+'.h5', custom_objects={'TCN':TCN})
        _self.network = tf.keras.models.load_model(filename+'.h5')
        return _self

    def fit(self, train_x, train_y, val_x, val_y, n_epochs=20, batch_size=60, shuffle=False, callback=None, verbose='auto'):
        self.name = self.name + '_e_' + str(n_epochs)
        self.network.fit(
            train_x,
            train_y,
            epochs=n_epochs,
            batch_size=batch_size,
            validation_data=(val_x, val_y),
            verbose=verbose,
            shuffle=shuffle,
            callbacks=callback
        )
        self.fitted = True


    def to_file(self, path_to_folder='./'):
        # Set Filename
        filename = os.path.join(path_to_folder, self.name)

        # Salvar rede Tensorflow
        self.network.save(filename+'.h5', overwrite=True)
        with open(filename+'.pkl', 'wb') as _file:
            _pickle.dump(self, _file, -1)
        
        print('Rede gravada em: ', filename)


    def __getstate__(self):
        return (self.input_shape, self.name, self.metrics, self.fitted)

    def __setstate__(self, state):
        self.input_shape, self.name, self.metrics, self.fitted = state
