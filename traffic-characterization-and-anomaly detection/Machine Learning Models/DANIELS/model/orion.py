import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef

sys.path.append('../')

from utils import *
from network import Network
from model_callback import Epoch_less_verbose
from plot_characterization import plot_prediction_real_graph

def get_attack_intervals(dataset_y:np.array):
    arr = dataset_y.copy()
    arr[arr != 0] = 1
    b = np.convolve(arr, np.array([-1, 1]))
    # Usar função convolve do numpy depois pegar o índice dos números 1s
    return np.concatenate([np.array(np.where(b == -1)[0]).reshape([-1, 1]),
                           np.array(np.where(b == 1)[0]).reshape([-1, 1])],
                           axis=1)

def load_models(segundos_de_batch:int, layers_type:list, layers_size:list,
base_name='', dropout=0, n_epochs=100, batch_size=240, output_function:str='tanh',
load_path:str=r'./modelos'):

    if kernel_size != None: base_name += f'k_{kernel_size}_'
    # ip_dst_s_5_t_32_d_16_d_1_e_20
    layers_type_size = '_'
    for layer, size in zip(layers_type, layers_size):
        layers_type_size += f'{layer}_{size}_'

    bytes_model    = Network.from_file(load_path + f'/bytes_{base_name}s_{segundos_de_batch}{layers_type_size}e_{n_epochs}')
    ip_dst_model   = Network.from_file(load_path + f'/ip_dst_{base_name}s_{segundos_de_batch}{layers_type_size}e_{n_epochs}')
    port_dst_model = Network.from_file(load_path + f'/port_dst_{base_name}s_{segundos_de_batch}{layers_type_size}e_{n_epochs}')
    ip_src_model   = Network.from_file(load_path + f'/ip_src_{base_name}s_{segundos_de_batch}{layers_type_size}e_{n_epochs}')
    port_src_model = Network.from_file(load_path + f'/port_src_{base_name}s_{segundos_de_batch}{layers_type_size}e_{n_epochs}')
    packets_model  = Network.from_file(load_path + f'/packets_{base_name}s_{segundos_de_batch}{layers_type_size}e_{n_epochs}')

    return bytes_model, ip_dst_model, port_dst_model, ip_src_model, port_src_model, packets_model

def fit_models(segundos_de_batch:int, layers_type:list, layers_size:list,
train_x:np.ndarray, train_y:np.ndarray, val_x:np.ndarray, val_y:np.ndarray,
base_name='', dropout=0, n_epochs=100, batch_size=360, output_function:str='tanh',
save_path:str=r'./modelos',
input_shape=None, kernel_size=None
):

    if input_shape == None: input_shape=(segundos_de_batch, 6)
    if kernel_size != None: base_name += f'k_{kernel_size}_'

    bytes_model = Network(input_shape=input_shape,
                            layers_type=layers_type,
                            layers_size=layers_size,
                            base_name='bytes_' +base_name,
                            dropout=dropout, output_function=output_function,
                            kernel_size=kernel_size
                        )

    bytes_model.fit(
        train_x,
        train_y[:, 0],
        val_x,
        val_y[:, 0],
        shuffle=True,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=1,
        # callback=[Epoch_less_verbose()]
    )
    bytes_model.to_file(save_path)



    ip_dst_model = Network(input_shape=input_shape,
                            layers_type=layers_type,
                            layers_size=layers_size,
                            base_name='ip_dst_' +base_name,
                            dropout=dropout, output_function=output_function,
                            kernel_size=kernel_size
                        )

    ip_dst_model.fit(
        train_x,
        train_y[:, 1],
        val_x,
        val_y[:, 1],
        shuffle=True,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=0,
        callback=[Epoch_less_verbose()]
    )
    ip_dst_model.to_file(save_path)


    port_dst_model = Network(input_shape=input_shape,
                            layers_type=layers_type,
                            layers_size=layers_size,
                            base_name='port_dst_' +base_name,
                            dropout=dropout, output_function=output_function,
                            kernel_size=kernel_size
                        )

    port_dst_model.fit(
        train_x,
        train_y[:, 2],
        val_x,
        val_y[:, 2],
        shuffle=True,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=0,
        callback=[Epoch_less_verbose()]
    )
    port_dst_model.to_file(save_path)



    ip_src_model = Network(input_shape=input_shape,
                            layers_type=layers_type,
                            layers_size=layers_size,
                            base_name='ip_src_' +base_name,
                            dropout=dropout, output_function=output_function,
                            kernel_size=kernel_size
                        )

    ip_src_model.fit(
        train_x,
        train_y[:, 3],
        val_x,
        val_y[:, 3],
        shuffle=True,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=0,
        callback=[Epoch_less_verbose()]
    )
    ip_src_model.to_file(save_path)



    port_src_model = Network(input_shape=input_shape,
                            layers_type=layers_type,
                            layers_size=layers_size,
                            base_name='port_src_' +base_name,
                            dropout=dropout, output_function=output_function,
                            kernel_size=kernel_size
                        )

    port_src_model.fit(
        train_x,
        train_y[:, 4],
        val_x,
        val_y[:, 4],
        shuffle=True,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=0,
        callback=[Epoch_less_verbose()]
    )
    port_src_model.to_file(save_path)



    packets_model = Network(input_shape=input_shape,
                            layers_type=layers_type,
                            layers_size=layers_size,
                            base_name='packets_' +base_name,
                            dropout=dropout, output_function=output_function,
                            kernel_size=kernel_size
                        )

    packets_model.fit(
        train_x,
        train_y[:, 5],
        val_x,
        val_y[:, 5],
        shuffle=True,
        n_epochs=n_epochs,
        batch_size=batch_size,
        verbose=0,
        callback=[Epoch_less_verbose()]
    )
    packets_model.to_file(save_path)



    return bytes_model, ip_dst_model, port_dst_model, ip_src_model, port_src_model, packets_model

def get_std_dev(janela_fuzzy, dataset:np.ndarray)->np.ndarray:

    std_dev = []

    # Caso especial quando não tem o suficiente para a janela
    for i in range(janela_fuzzy-1):
        std_dev.append(np.std(dataset[0:i+1, :], axis=0))
    # Caso normal até o final
    # [0 1 2 3 4 5 6 7 8 9] --> # Para referência
    for i, j in zip(range(0, dataset.shape[0]+2-janela_fuzzy , 1), range(janela_fuzzy, dataset.shape[0]+1, 1)):
        std_dev.append(np.std(dataset[i:j, :], axis=0))

    std_dev = np.array(std_dev)
    std_dev[std_dev == 0] = 0.000001
    return std_dev

def fuzzyfy(error:np.ndarray, std_dev:np.ndarray)->np.ndarray:
    fuzz = np.sum(
                -np.expm1(
                    # (-np.power(error, 2))/(2*np.power(std_dev, 2))
                    # (-np.power(error, 2))/(2*(4.47**2))
                    (-np.power(error, 2))/(2*np.power(std_dev[:, 1:5]*4.47, 2))
                    
                )
            , axis=1)

    return fuzz

def get_fuzzy_threshold(models:list, dataset_x:np.ndarray, dataset_y:np.ndarray, labels:np.ndarray, std_dev:np.ndarray)->float:
    labels[labels == 2] = 1

    prediction = np.concatenate((
        # models[0].network.predict(dataset_x),
        models[1].network.predict(dataset_x),
        models[2].network.predict(dataset_x),
        models[3].network.predict(dataset_x),
        models[4].network.predict(dataset_x),
        # models[5].network.predict(dataset_x)
    ), axis=1)

    error = prediction - dataset_y[:, 1:5]

    ans_fuzzy = fuzzyfy(error, std_dev)

    # Calculando Melhor limiar fuzzy
    print('Calculando melhor limiar fuzzy...')
    best_matthew_coef = -1
    best_treshold=0
    f1_results = []
    matthew_results = []
    # I = np.arange(0.01, 6.7, 0.01)
    # interval = np.arange(0.0001, 0.8, 0.001)
    interval = np.arange(0.0001, 2, 0.005)
    for lim in tqdm(interval):
        fuzzy:np.ndarray = np.zeros(ans_fuzzy.shape)
        fuzzy[ans_fuzzy > lim] = 1
        try:

            matthew_coef = matthews_corrcoef(labels, fuzzy)
            f1 = f1_score(labels, fuzzy)
        except ZeroDivisionError:
            matthew_coef = -1

        f1_results.append(f1)
        matthew_results.append(matthew_coef)

        if matthew_coef > best_matthew_coef:
            best_matthew_coef = matthew_coef
            best_f1 = f1
            best_treshold = lim

    print('Melhor limiar fuzzy: ', best_treshold)
    print('Mathew coeficient do melhor threshold: ', best_matthew_coef)
    print('f1 do threshold com melhor mathew coeficient: ', best_f1)

    if not sys.argv.count('-noplot'):

        # Plot F1 ao longo dos testes
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(interval, f1_results)
        ax.set(xlabel='Limiares', ylabel='F1', title='Fuzzy Thresholds')
        ax.vlines(best_treshold, 0, 1, colors=['k'])
        # plt.ion()

        # Plot mathew coeficient ao longo dos testes
        fig, ax = plt.subplots(constrained_layout=True)
        ax.plot(interval, matthew_results)
        ax.set(xlabel='Limiares', ylabel='Matthew Coeficient', title='Fuzzy Thresholds')
        ax.vlines(best_treshold, -1, 1, colors=['k'])

        # plt.show(block=False)


    fuzzy:np.ndarray = np.zeros(ans_fuzzy.shape)
    fuzzy[ans_fuzzy > best_treshold] = 1
    print(confusion_matrix(labels, fuzzy))

    return best_treshold

def test_model(models:list, dataset_x:np.ndarray, dataset_y:np.ndarray, labels:np.ndarray, std_dev:np.ndarray, fuzzy_treshold:float, models_path:str):
    labels[labels == 2] = 1
    prediction = np.concatenate((
        models[0].network.predict(dataset_x),
        models[1].network.predict(dataset_x),
        models[2].network.predict(dataset_x),
        models[3].network.predict(dataset_x),
        models[4].network.predict(dataset_x),
        models[5].network.predict(dataset_x)
    ), axis=1)

    error = prediction[:, 1:5] - dataset_y[:, 1:5]
    ans_fuzzy = fuzzyfy(error, std_dev)

    model_ans = np.zeros(labels.shape)
    model_ans[ans_fuzzy > fuzzy_treshold] = 1
    print()
    print('Resultados teste:')
    print('precision teste: ', precision_score(labels, model_ans))
    print('recall teste: ', recall_score(labels, model_ans))
    print('F1 teste: ', f1_score(labels, model_ans))
    print('Matthew Coef: ', matthews_corrcoef(labels, model_ans))
    print(confusion_matrix(labels, model_ans))

    if not sys.argv.count('-noplot'):
        plot_prediction_real_graph('Dia de teste',
                                    models_path+'/plots/dia_teste.jpg',
                                    prediction,
                                    dataset_y,
                                    get_attack_intervals(labels)
                                    )

if __name__ == '__main__':
    # SEGUNDOS_DE_BATCH = 6
    SEGUNDOS_DE_BATCH = 10
    fuzzy_window = 50
    # Network Params:
    seconds = SEGUNDOS_DE_BATCH
    
    # CNN
    # layer_type = ['c', 'm', 'f', 'd', 'd']
    # layer_size = [32, 2, 0, 16, 1]
    # kernel_size = [3]
    # n_epochs = 100
    
    # TCN
    # layer_type = ['t', 'd', 'd']
    # layer_size = [32, 16, 1]
    # kernel_size = [3]
    # dilations=[1, 2, 3]
    # n_epochs = 100


    # GRU
    layer_type = ['g', 'd', 'd']
    layer_size = [32, 16, 1]
    # layer_type = ['g', 'd']
    # layer_size = [32, 1]
    n_epochs = 100
    output_function=None
    kernel_size = None

    # Input precisa ser formatado diferente para cnn
    cnn = True if layer_type[0] == 'c' else False
    input_shape = (SEGUNDOS_DE_BATCH, 6, 1) if cnn else None

    # import data
    if sys.argv.count('-ugr'):
        dia_sem_ataque, dia_1_ataque, dia_2_ataque = import_data_ugr()
        models_path = r'./modelos/ugr'
    elif sys.argv.count('-cic'):
        dia_sem_ataque, dia_1_ataque, dia_2_ataque = import_data_cic()
        models_path = r'./modelos/cic'
    elif sys.argv.count('-old'):
        dia_sem_ataque, dia_1_ataque, dia_2_ataque = import_data_old()
        models_path = r'./modelos/old'
    else:
        dia_sem_ataque, dia_1_ataque, dia_2_ataque = import_data_globecom()
        models_path = r'./modelos/orion'

    label_safe = dia_sem_ataque[seconds+1:, 6],
    label_1 = dia_1_ataque[seconds+1:, 6]
    label_2 = dia_2_ataque[seconds+1:, 6]
    

    # Pegar std_dev do dataset antes de cortar os segundos dos dados,  depois cortar
    std_dev_safe = get_std_dev(fuzzy_window, dia_sem_ataque)[seconds +1:, :6]
    std_dev_1 = get_std_dev(fuzzy_window, dia_1_ataque)[seconds +1:, :6]
    std_dev_2 = get_std_dev(fuzzy_window, dia_2_ataque)[seconds +1:, :6]


    # Organize data
    dia_sem_ataque, input_dia_sem_ataque = stack_seconds(dia_sem_ataque[:, :6], SEGUNDOS_DE_BATCH)
    dia_1_ataque, input_dia_1_ataque = stack_seconds(dia_1_ataque[:, :6], SEGUNDOS_DE_BATCH)
    dia_2_ataque, input_dia_2_ataque = stack_seconds(dia_2_ataque[:, :6], SEGUNDOS_DE_BATCH)

    input_dia_sem_ataque, dia_sem_ataque = fix_data(input_dia_sem_ataque, dia_sem_ataque)
    input_dia_1_ataque, dia_1_ataque = fix_data(input_dia_1_ataque, dia_1_ataque)
    input_dia_2_ataque, dia_2_ataque = fix_data(input_dia_2_ataque, dia_2_ataque)


    if cnn:
        input_dia_sem_ataque = input_dia_sem_ataque.reshape(input_dia_sem_ataque.shape[0], SEGUNDOS_DE_BATCH, 6, 1)
        input_dia_1_ataque = input_dia_1_ataque.reshape(input_dia_1_ataque.shape[0], SEGUNDOS_DE_BATCH, 6, 1)
        input_dia_2_ataque = input_dia_2_ataque.reshape(input_dia_2_ataque.shape[0], SEGUNDOS_DE_BATCH, 6, 1)

    # Separar treino e validação
    train_x, val_x, train_y, val_y = \
    train_test_split(input_dia_sem_ataque, dia_sem_ataque, test_size=0.2, random_state=42)


    if sys.argv.count('-t'):
        # Treinar modelos
        bytes_model,    \
        ip_dst_model,   \
        port_dst_model, \
        ip_src_model,   \
        port_src_model, \
        packets_model = \
        fit_models(seconds, layer_type, layer_size, train_x, train_y, val_x, val_y,
                n_epochs=n_epochs, save_path=models_path, input_shape=input_shape,
                kernel_size=kernel_size, output_function=output_function)
    else:
        # Carregar modelos
        bytes_model,    \
        ip_dst_model,   \
        port_dst_model, \
        ip_src_model,   \
        port_src_model, \
        packets_model = \
        load_models(seconds, layer_type, layer_size, n_epochs=n_epochs, load_path=models_path)

    models = [bytes_model, ip_dst_model, port_dst_model, ip_src_model, port_src_model, packets_model]

    fuzz_treshold = get_fuzzy_threshold(models, input_dia_1_ataque, dia_1_ataque, label_1, std_dev_1)

    test_model(models, input_dia_2_ataque, dia_2_ataque, label_2, std_dev_2, fuzz_treshold, models_path)

    plt.show()
    # input('Press enter to exit...')