import numpy as np
from keras.models import load_model
from MLP import LaymanMLP, TrainedMPL


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
        for i in range(0, 6):
            prediction = self.models[i].predict(x)
            absolute_error_sum += abs(prediction - y[:, i].reshape(-1, 1))

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
        answer = np.zeros((86380, 1))
        absolute_error_sum = 0
        for i in range(0, 6):
            prediction = self.models[i].predict(x)
            absolute_error_sum += abs(prediction - y[:, i].reshape(-1, 1))

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
