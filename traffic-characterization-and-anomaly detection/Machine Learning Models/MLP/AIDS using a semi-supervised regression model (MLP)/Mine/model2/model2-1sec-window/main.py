#
# Modelo semi-supervisionado de regressao que implementa a parte principal de um
# Anomaly-based Intrusion Detection System (AIDS).
#
# Os dados de entrada desse sub-sistema são atributos de fluxos IP da rede coletados ao longo de 1 segundo. Por exemplo,
# a soma de bytes e pacotes de todos os fluxos ip e a entropia do ip de origem de todos os fluxos observados em
# um certo segundo. O conjunto de dados de treinamento/validacao/teste, portanto, sera os atributos dos
# fluxos IP da rede, coletados segundo a segundo ao longo de um certo intervalo de tempo (geralmente um dia inteiro).
#
# O modelo, inicialmente, aprendera o perfil de comportamento normal do trafego da rede (em um dia sem ataques), e,
# e sera definido limites (threshold) para a variação do comportamento da rede.
# Em seguida, baseando-se no comportamento normal aprendido atraves de dados historicos da rede, o sistema pode começar
# a fazer previsões sobre o comportamento esperado da rede em diferentes intervalos de um certo dia.
# Com essas previsões em mãos, ele as compara com o comportamento real da rede e caso eles sejam muito discrepantes
# (comportamento real fora do threshold) um alarme é soado para o gerente da rede, indicando uma anomalia.
#


from numpy.core.numeric import zeros_like
import numpy as np
from sklearn.metrics import confusion_matrix

import tensorflow as tf
from MLP import LaymanMLP, TrainedMPL
from AnomalyDetector import LaymanAnomalyDetector, TrainedAnomalyDetector

import os
import sys
sys.path.insert(1, '../../../../')
from ImportTrafficData import ImportTrafficData

#
## Passo 0, garantir que nao existe nenhum arquivo de scalers proveniente de execucoes anteriores:
#
try:
    os.remove('scalers.pkl')
except FileNotFoundError:
    pass

#
## Passo 1, importar e preprocessar os dados:
#
normal_day_x, normal_day_y,\
normal_day_labels = ImportTrafficData.in_regression_model_format('../../../../../../Data/orion/051218_60h6sw_c1_ht5_it0_V2_csv', [])


day_1_x, day_1_y,\
day_1_labels = ImportTrafficData.in_regression_model_format('../../../../../../Data/orion/051218_60h6sw_c1_ht5_it0_V2_csv_ddos_portscan',
                                                            # Os ataques acontecem nesses horários:
                                                            [('10:15:00', '11:30:00'),
                                                            ('13:25:00', '14:35:00')])

# Vamos filtrar os segundos sem ataque do dia 1 para usar como conjunto de validacao durante o treinamento.
day_1_x = day_1_x[day_1_labels == 0, :]  #accessing elements in numpy ndarrays: [start:end:step, start:end:step, ...]
day_1_y = day_1_y[day_1_labels == 0, :]
day_1_labels = day_1_labels[day_1_labels == 0]


day_2_x, day_2_y,\
day_2_labels = ImportTrafficData.in_regression_model_format('../../../../../../Data/orion/171218_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos',
                                                            [('09:45:00', '11:10:00'),
                                                            ('17:37:00', '18:55:00')])


day_3_x, day_3_y,\
day_3_labels = ImportTrafficData.in_regression_model_format('../../../../../../Data/orion/120319_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos',
                                                            [('09:29:00', '10:44:00'),
                                                            ('16:16:00', '17:50:00')])


#
## Passo 2, Criar/importar um modelo de regressão do tensorflow.
# Vamos utilizar um modelo/threshold diferente para realizar a previsao de cada atributo do proximo segundo.
#
#
# Se o usuario executar o programa passando o argumento -t vamos configurar e treinar (criar) o modelo.
if sys.argv.count("-t"):
    model = LaymanAnomalyDetector((6, 16, 1), ("sigmoid", None), "mse", "adam", ["mse"])
    model.save("./model")

# Se nao, vamos importar um modelo ja previamente configurado e treinado.
else:
    model = TrainedAnomalyDetector("./model")


#
## Passo 3, encontrar thresholds que garantam o melhor f1 score para o modelo:
#
model.set_thresholds(day_2_x, day_2_y, day_2_labels)

#
## Passo 4, Testar o modelo em um novo dia COM anomalias:
#
answer = model.detect(day_3_x, day_3_y)

precision, recall, f1 = model.evaluate_performance(answer, day_3_labels)
print('\n\nMetrics:')
print("Confusion matrix: ")
tn, fp, fn, tp = confusion_matrix(day_3_labels, answer).ravel()
print(np.array([[tp, fp], [fn, tn]]))
print('Precision: ', precision)
print('Recall: ', recall)
print('F1: ', f1)