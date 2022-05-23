import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, GRU
from keras import callbacks
import matplotlib.pyplot as plt


def plot_training_metrics_graph(history, metric, model):
    plt.figure(figsize = (10, 6))
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric])
    plt.title(f'{model}th model: train {metric} vs validation {metric}')
    plt.xlabel("epoch")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric], loc='upper right')
    plt.tight_layout()
    plt.savefig("./model_"+str(model)+metric+".jpg", format='jpg', dpi=800)


class Gru:
    def __init__(self):
        pass

    def print_model_organization(self):
        self.model.summary()

    def predict(self, x: list) -> list:
        return self.model.predict(x)

from keras import backend as K

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


class LaymanGru(Gru):
    def __init__(self, input_shape: tuple, layers: tuple, loss: str, optimizer: str, metrics: list, dropout=0.0):
        self.model = Sequential()
        self.model.add(GRU(units=layers[0], activation="tanh", recurrent_activation="sigmoid", input_shape=[input_shape[0], input_shape[1]]))
        if dropout > 0.0:
            self.model.add(Dropout(dropout))
        self.model.add(Dense(units=layers[1], activation="sigmoid"))
        self.model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy', precision_m, recall_m])

        
    def train(self, x, y, val_x, val_y, batch_size=64, epochs=10, model=""):

        early_stop = None
        if (val_x is not None) and (val_y is not None):
            early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10)

        history = self.model.fit(
            x=x,
            y=y,
            validation_data=(val_x, val_y),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[early_stop]
        )
        
        if (val_x is not None) and (val_y is not None):
            plot_training_metrics_graph(history, 'loss', model)
            plot_training_metrics_graph(history, 'accuracy', model)


    def save(self, path: str):
        self.model.save(path)


class TrainedGru(Gru):
    def __init__(self, path: str):
        self.model = load_model(path)
