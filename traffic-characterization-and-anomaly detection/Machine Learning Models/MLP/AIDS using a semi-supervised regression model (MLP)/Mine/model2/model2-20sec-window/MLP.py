import numpy as np
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout


class MLP:
    def __init__(self):
        pass

    def print_model_organization(self):
        self.model.summary()

    def predict(self, x: list) -> list:
        return self.model.predict(x)


class LaymanMLP(MLP):
    def __init__(self, layers: tuple, activations: tuple, loss: str, optimizer: str, metrics: list, dropout=0.0):
        self.model = Sequential()
        self.model.add(Dense(layers[1], input_dim=layers[0], activation=activations[0]))
        if dropout > 0.0:
            self.model.add(Dropout(dropout))
        self.model.add(Dense(layers[2], activation=activations[1]))
        self.model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def train(self, x, y, val_x, val_y, batch_size=60, epochs=10):
        self.model.fit(
            x=x,
            y=y,
            validation_data=(val_x, val_y),
            batch_size=batch_size,
            epochs=epochs
        )

    def save(self, path: str):
        self.model.save(path)


class TrainedMPL(MLP):
    def __init__(self, path: str):
        self.model = load_model(path)
