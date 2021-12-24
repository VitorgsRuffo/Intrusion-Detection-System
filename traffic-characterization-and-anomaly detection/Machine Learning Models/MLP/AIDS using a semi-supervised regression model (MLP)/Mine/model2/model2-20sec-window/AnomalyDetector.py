import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from keras.models import load_model
from MLP import LaymanMLP, TrainedMPL
import sys
sys.path.insert(1, '../../../../')
from ImportTrafficData import ImportTrafficData

def plot_prediction_real_graph_orion(graph_name: str, save_path: str, predicted_data, real_data, attack_intervals: list):
    features_names = ['bytes', 'dst_ip_entropy', 'dst_port_entropy',
                      'src_ip_entropy', 'src_port_entropy', 'packets']
    mpl.rcParams['lines.linewidth'] = 0.5
    #mpl.style.use('seaborn') #['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
    plt.margins(x=0)
    figure, plots = plt.subplots(3, 2, constrained_layout=True, figsize=(10.80,7.20))
    plt.suptitle(graph_name, fontsize=14)

    lin_space = np.linspace(0, 24, 86380)
    for k in range(0, 6):
        i = k // 2 # mapping array index to matrix indices. (obs: 2 == matrix line length)
        j = k % 2
        plots[i][j].set_title(features_names[k])
        plots[i][j].step(lin_space, real_data[:, k], label='REAL', color='green')
        plots[i][j].step(lin_space, predicted_data[:, k], label='PREDICTED', color='orange')
        plots[i][j].set_xticks(np.arange(0, 25, 2))
        plots[i][j].margins(x=0)
        plots[i][j].legend(fontsize='xx-small')
        mse = round(mean_squared_error(real_data[:, k], predicted_data[:, k]), 6)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plots[i][j].text(0.05, 0.90, f"MSE = {mse}", 
            transform=plots[i][j].transAxes, fontsize=7, bbox=props) 
        
        for attack_interval in attack_intervals:
            start, end = \
                ImportTrafficData.time_to_seconds(attack_interval[0]), ImportTrafficData.time_to_seconds(attack_interval[1])
            start, end = lin_space[start], lin_space[end]
            plots[i][j].axvspan(start, end, label="ATTACK", color='r', alpha=0.2)

        if k == 0:
            #plots[i][j].set_ylim([0, x])
            plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        elif k == 5:
            #plots[i][j].set_ylim([0, y])
            plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        else:
            plots[i][j].set_ylim([-25, 15])

    plt.tight_layout()
    plt.savefig(save_path, format='jpg', dpi=800)

def plot_prediction_real_graph_ugr16(graph_name: str, save_path: str, predicted_data, real_data):
    features_names = ['bytes', 'dst_ip_entropy', 'dst_port_entropy',
                      'src_ip_entropy', 'src_port_entropy', 'packets']
    mpl.rcParams['lines.linewidth'] = 0.5
    #mpl.style.use('seaborn') #['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
    plt.margins(x=0)
    figure, plots = plt.subplots(3, 2, figsize=(10.80,7.20))
    plt.suptitle(graph_name, fontsize=14)

    lin_space = np.linspace(0, 24, 86380)
    for k in range(0, 6):
        i = k // 2 # mapping array index to matrix indices. (obs: 2 == matrix line length)
        j = k % 2
        plots[i][j].set_title(features_names[k])
        plots[i][j].step(lin_space, real_data[:, k], label='REAL', color='green')
        plots[i][j].step(lin_space, predicted_data[:, k], label='PREDICTED', color='orange')
        plots[i][j].set_xticks(np.arange(0, 25, 2))
        plots[i][j].margins(x=0)
        plots[i][j].legend(fontsize='xx-small')
        mse = round(mean_squared_error(real_data[:, k], predicted_data[:, k]), 6)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plots[i][j].text(0.05, 0.90, f"MSE = {mse}", 
            transform=plots[i][j].transAxes, fontsize=7, bbox=props) 

        if k == 0:
            #plots[i][j].set_ylim([0, 3000000000])
            #plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            pass
        elif k == 5:
            #plots[i][j].set_ylim([0, 2500000])
            #plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
            pass
        else:
            plots[i][j].set_ylim([-20, 11])

    plt.tight_layout()
    plt.savefig(save_path, format='jpg', dpi=800)


def get_f1(ans: np.ndarray, labels: np.ndarray, all_metrics=False):
    # é possível otimizar esse código com o numpy,
    # mas dá muito trabalho e já é rápido o suficiente.
    # Em um outro dia eu mostro como
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    for second, label in zip(ans, labels):
        if second == label:
            if label == 1:
                true_positive += 1
            else:
                true_negative += 1
        else:
            if label == 1:
                false_negative += 1
            else:
                false_positive += 1

    try:
        precision = true_positive / (true_positive + false_positive)
    except ZeroDivisionError:  # Caso não exista uma amostra dessa classe
        precision = 1.0

    try:
        recall = true_positive / (true_positive + false_negative)
    except ZeroDivisionError:  # Caso não exista uma amostra dessa classe
        recall = 1.0

    f1 = 2 * ((precision * recall) / (precision + recall))



    if all_metrics:
        return precision, recall, f1

    return f1


class AnomalyDetector:
    def __init__(self):
        self.models = [None, None, None, None, None, None]
        self.threshold = -1

    def set_threshold(self, x, y, labels):
        absolute_error_sum = 0
        predictions = np.zeros((86380, 1)) #variavel para salvar as previsoes em formato de matriz

        for i in range(0, 6):
            prediction = self.models[i].predict(x)
            predictions = np.column_stack((predictions, prediction))
            absolute_error_sum += abs(prediction - y[:, i].reshape(-1, 1))

        predictions = predictions[:, 1:] #removendo a coluna nula.
        # plot_prediction_real_graph("Prediction-real graph:\n171218_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos - 6 MLPs with a 1-neuron output layer", 
        #                             "./171218_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos_6mlp1nol_prediction_real_graph.jpg",
        #                             predictions, y, [('09:45:00', '11:10:00'), ('17:37:00', '18:55:00')])


        best_f1_score = -1
        best_threshold = -1
        interval = np.linspace(0.0, 10.0, num=100)
        for threshold in interval:
            # Ans é um array de 86399 posicoes que representa a resposta do modelo.
            # obs: lembrar que o modelo começa a prever a partir do segundo segundo do dia por isso
            # são apenas 86399 posicoes.
            # Se ans[i] = 0, segundo o modelo, o i-esimo segundo é normal.
            # Se ans[i] = 1, segundo o modelo, o i-esimo segundo é anomalo.
            ans = np.zeros_like(absolute_error_sum)

            # Os segundos nos quais a soma do erro dos atributos é maior que o threshold sao marcados como anomalos:
            ans[absolute_error_sum > threshold] = 1

            # Calcula o f1 score considerando as respostas atuais do modelo e as respostas corretas:
            f1 = get_f1(ans, labels)

            # Se esse f1 score for melhor que o melhor f1 calculado anteriormente
            # vamos salvar o threshold atual e esse novo f1.
            if f1 > best_f1_score:
                best_f1_score = f1
                best_threshold = threshold

        #salvando o threshold encontrado para a variacao dos atributos de um segundo normal.
        self.threshold = best_threshold

    def detect(self, x, y) -> list:
        # Vamos usar um modelo diferente para prever cada um dos atributos.
        # Vamos utilizar um threshold que indica maximo que os atributos de um
        # segundo podem variar em conjunto para ele ser considerado como normal.
        predictions = np.zeros((86380, 1)) #variavel para salvar as previsoes em formato de matriz
        answer = np.zeros((86380, 1))
        absolute_error_sum = 0
        for i in range(0, 6):
            prediction = self.models[i].predict(x)
            predictions = np.column_stack((predictions, prediction))
            absolute_error_sum += abs(prediction - y[:, i].reshape(-1, 1))
        
        predictions = predictions[:, 1:] #removendo a coluna nula.
        # plot_prediction_real_graph_orion("Prediction-real graph:\n051218_60h6sw_c1_ht5_it0_V2_csv_ddos_portscan - 6 MLPs with a 1-neuron output layer", 
        #                            "./051218_60h6sw_c1_ht5_it0_V2_csv_ddos_portscan_6mlp1nol_prediction_real_graph.jpg",
        #                            predictions, y, [('10:15:00', '11:30:00'), ('13:25:00', '14:35:00')])
        # plot_prediction_real_graph_orion("Prediction-real graph:\n120319_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos - 6 MLPs with a 1-neuron output layer", 
        #                            "./120319_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos_6mlp1nol_prediction_real_graph.jpg",
        #                            predictions, y, [('09:29:00', '10:44:00'), ('16:16:00', '17:50:00')])
        # plot_prediction_real_graph_ugr16("Prediction-real graph:\npure_training_day_10 - 6 MLPs with a 1-neuron output layer", 
        #                                  "./pure_training_day_10_6mlp1nol_prediction_real_graph(trained-with-8-9).jpg",
        #                                  predictions, y)
        plot_prediction_real_graph_ugr16("Prediction-real graph:\npure_training_day_7 - 6 MLPs with a 1-neuron output layer", 
                                         "./pure_training_day_7_6mlp1nol_prediction_real_graph(trained-with-3-4-5-6).jpg",
                                         predictions, y)

        answer[absolute_error_sum > self.threshold] = 1
        return answer

    def evaluate_performance(self, answer, labels):
        return get_f1(answer, labels, all_metrics=True)


class LaymanAnomalyDetector(AnomalyDetector):
    def __init__(self, layer: tuple, activation: tuple, loss: str, optimizer: str, metrics: list, dropout=0.0):
        super().__init__()
        bytes_model        = LaymanMLP(layer, activation, loss, optimizer, metrics, dropout)
        dst_ip_ent_model   = LaymanMLP(layer, activation, loss, optimizer, metrics, dropout)
        dst_port_ent_model = LaymanMLP(layer, activation, loss, optimizer, metrics, dropout)
        src_ip_ent_model   = LaymanMLP(layer, activation, loss, optimizer, metrics, dropout)
        src_port_ent_model = LaymanMLP(layer, activation, loss, optimizer, metrics, dropout)
        packets_model      = LaymanMLP(layer, activation, loss, optimizer, metrics, dropout)
        self.models = [bytes_model, dst_ip_ent_model, dst_port_ent_model,
                       src_ip_ent_model, src_port_ent_model, packets_model]

    def train(self, x, y, val_x, val_y, batch_size=60, epochs=10):
        for i in range(0, 6):
            print("\n\nTraining {}th model...".format(i))
            self.models[i].train(x, y[:, i], val_x, val_y[:, i], batch_size, epochs)

    def save(self, path: str):
        for i in range(0, 6):
            self.models[i].save(path+"-"+str(i))


class TrainedAnomalyDetector(AnomalyDetector):
    def __init__(self, path: str):
        super().__init__()
        for i in range(0, 6):
            self.models[i] = TrainedMPL(path+"-"+str(i))
