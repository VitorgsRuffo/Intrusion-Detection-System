from tensorflow.keras.callbacks import Callback
class Model_Callback(Callback):
    def __init__(self):
        self.loss_epoch = []

    def on_epoch_end(self, epoch, logs):
        self.loss_epoch.append(logs['val_loss'])
        # return super().on_epoch_end(epoch, logs=logs)




class Epoch_less_verbose(Callback):
    def __init__(self):
        # self.loss_epoch = []
        pass

    def on_epoch_end(self, epoch, logs):
        if epoch % 10 == 9 or epoch == 0:
            print(f'Epoch: {epoch}, mse: {logs["mse"]}, val_mse: {logs["val_mse"]}')

        # if epoch % 10 == 0:
        #     self.params['verbose'] = 0
        #     self.set_params({'verbose':1})
        # else:
        #     self.params['verbose'] = 0
        #     self.set_params({'verbose':0})
        # return super().on_epoch_end(epoch, logs=logs)