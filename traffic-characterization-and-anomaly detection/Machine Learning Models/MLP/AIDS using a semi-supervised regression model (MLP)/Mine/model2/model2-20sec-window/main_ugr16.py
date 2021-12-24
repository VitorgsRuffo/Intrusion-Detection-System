#
# Modelo semi-supervisionado de regressao que implementa a parte principal de um
# Anomaly-based Intrusion Detection System (AIDS).
#
# Os dados de entrada desse IDS são atributos de fluxos IP da rede coletados ao longo de 1 segundo. Por exemplo,
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
# normal_day_x, normal_day_y,\
# normal_day_labels = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../../Data/ugr16/training/pure_training_day_8.csv')

# normal_day2_x, normal_day2_y,\
# normal_day2_labels = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../../Data/ugr16/training/pure_training_day_9.csv')

# normal_day3_x, normal_day3_y,\
# normal_day3_labels = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../../Data/ugr16/training/pure_training_day_10.csv')


# normal_day_x = np.row_stack((normal_day_x, normal_day2_x))
# normal_day_y = np.row_stack((normal_day_y, normal_day2_y))
# normal_day_labels = np.row_stack((normal_day_labels, normal_day2_labels))

normal_day_x, normal_day_y,\
normal_day_labels = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../../Data/ugr16/training/pure_training_day_3.csv')

normal_day2_x, normal_day2_y,\
normal_day2_labels = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../../Data/ugr16/training/pure_training_day_4.csv')

normal_day3_x, normal_day3_y,\
normal_day3_labels = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../../Data/ugr16/training/pure_training_day_5.csv')

normal_day4_x, normal_day4_y,\
normal_day4_labels = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../../Data/ugr16/training/pure_training_day_6.csv')

normal_day5_x, normal_day5_y,\
normal_day5_labels = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../../Data/ugr16/training/pure_training_day_7.csv')


normal_day_x = np.row_stack((normal_day_x, normal_day2_x, normal_day3_x, normal_day4_x))
normal_day_y = np.row_stack((normal_day_y, normal_day2_y, normal_day3_y, normal_day4_y))
normal_day_labels = np.row_stack((normal_day_labels, normal_day2_labels, normal_day3_labels, normal_day4_labels))



# attack days.
# day_2_x, day_2_y,\
# day_2_labels = ImportTrafficData.in_regression_model_20_sec_format('../../../../../../Data/orion/171218_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos',
#                                                             [('09:45:00', '11:10:00'),
#                                                             ('17:37:00', '18:55:00')])


# day_3_x, day_3_y,\
# day_3_labels = ImportTrafficData.in_regression_model_20_sec_format('../../../../../../Data/orion/120319_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos',
#                                                             [('09:29:00', '10:44:00'),
#                                                             ('16:16:00', '17:50:00')])


#
## Passo 2, Criar/importar um modelo de regressão do tensorflow.
# Vamos utilizar um modelo/threshold diferente para realizar a previsao de cada atributo do proximo segundo.
#
#
# Se o usuario executar o programa passando o argumento -t vamos configurar e treinar (criar) o modelo.
if sys.argv.count("-t"):
    model = LaymanAnomalyDetector((120, 16, 1), ("sigmoid", None), "mse", "adam", ["mse"])
    model.train(normal_day_x, normal_day_y, normal_day5_x, normal_day5_y)
    # 2.1, encontrar um threshold que garantam o melhor f1 score para o modelo (em um dia qualquer com ataques):
    #model.set_threshold(day_2_x, day_2_y, day_2_labels)
    model.save("./model")

# Se nao, vamos importar um modelo ja previamente configurado e treinado.
else:
    model = TrainedAnomalyDetector("./model")

#
## Passo 3, Testar o modelo em dias COM anomalias:
#

# answer = model.detect(day_1_x, day_1_y)
# precision, recall, f1 = model.evaluate_performance(answer, day_1_labels)
# print('\n\nDay 1 Metrics:')
# print("Confusion matrix: ")
# tn, fp, fn, tp = confusion_matrix(day_1_labels, answer).ravel()
# print(np.array([[tp, fp], [fn, tn]]))
# print('Precision: ', precision)
# print('Recall: ', recall)
# print('F1: ', f1)


answer = model.detect(normal_day5_x, normal_day5_y)
# precision, recall, f1 = model.evaluate_performance(answer, day_3_labels)
# print('\n\nDay 3 Metrics:')
# print("Confusion matrix: ")
# tn, fp, fn, tp = confusion_matrix(day_3_labels, answer).ravel()
# print(np.array([[tp, fp], [fn, tn]]))
# print('Precision: ', precision)
# print('Recall: ', recall)
# print('F1: ', f1)
