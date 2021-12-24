
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

import os
import sys
from pickle import dump, load

sys.path.insert(1, '../../../')
from ImportTrafficData import ImportTrafficData


def plot_prediction_real_graph(graph_name: str, save_path: str, predicted_data, real_data):
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


# treinando com 4 dias
# dia_sem_ataque_x, dia_sem_ataque_y,\
# dia_sem_ataque_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_3.csv')

# dia_sem_ataque2_x, dia_sem_ataque2_y,\
# dia_sem_ataque2_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_4.csv')


# dia_sem_ataque3_x, dia_sem_ataque3_y,\
# dia_sem_ataque3_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_5.csv')

# dia_sem_ataque4_x, dia_sem_ataque4_y,\
# dia_sem_ataque4_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_6.csv')

# dia_sem_ataque5_x, dia_sem_ataque5_y,\
# dia_sem_ataque5_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_7.csv')


# dia_sem_ataque_x = np.row_stack((dia_sem_ataque_x, dia_sem_ataque2_x, dia_sem_ataque3_x, dia_sem_ataque4_x))
# dia_sem_ataque_y = np.row_stack((dia_sem_ataque_y, dia_sem_ataque2_y, dia_sem_ataque3_y, dia_sem_ataque4_y))
# dia_sem_ataque_rotulo = np.row_stack((dia_sem_ataque_rotulo, dia_sem_ataque2_rotulo, dia_sem_ataque3_rotulo, dia_sem_ataque4_rotulo))


## treinando com 2 dias
dia_sem_ataque_x, dia_sem_ataque_y,\
dia_sem_ataque_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_3.csv')

dia_sem_ataque2_x, dia_sem_ataque2_y,\
dia_sem_ataque2_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_4.csv')


dia_sem_ataque3_x, dia_sem_ataque3_y,\
dia_sem_ataque3_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_5.csv')

dia_sem_ataque_x = np.row_stack((dia_sem_ataque_x, dia_sem_ataque2_x))
dia_sem_ataque_y = np.row_stack((dia_sem_ataque_y, dia_sem_ataque2_y))
dia_sem_ataque_rotulo = np.row_stack((dia_sem_ataque_rotulo, dia_sem_ataque2_rotulo))




## treinando com 2 dias
# dia_sem_ataque_x, dia_sem_ataque_y,\
# dia_sem_ataque_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_8.csv')

# dia_sem_ataque2_x, dia_sem_ataque2_y,\
# dia_sem_ataque2_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_9.csv')


# dia_sem_ataque3_x, dia_sem_ataque3_y,\
# dia_sem_ataque3_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_10.csv')

# dia_sem_ataque_x = np.row_stack((dia_sem_ataque_x, dia_sem_ataque2_x))
# dia_sem_ataque_y = np.row_stack((dia_sem_ataque_y, dia_sem_ataque2_y))
# dia_sem_ataque_rotulo = np.row_stack((dia_sem_ataque_rotulo, dia_sem_ataque2_rotulo))



# treinando com 3 dias
# dia_sem_ataque_x, dia_sem_ataque_y,\
# dia_sem_ataque_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_8.csv')

# dia_sem_ataque2_x, dia_sem_ataque2_y,\
# dia_sem_ataque2_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_7.csv')


# dia_sem_ataque3_x, dia_sem_ataque3_y,\
# dia_sem_ataque3_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_9.csv')

# dia_sem_ataque4_x, dia_sem_ataque4_y,\
# dia_sem_ataque4_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/training/pure_training_day_10.csv')


# dia_sem_ataque_x = np.row_stack((dia_sem_ataque_x, dia_sem_ataque3_x, dia_sem_ataque4_x))
# dia_sem_ataque_y = np.row_stack((dia_sem_ataque_y, dia_sem_ataque3_y, dia_sem_ataque4_y))
# dia_sem_ataque_rotulo = np.row_stack((dia_sem_ataque_rotulo, dia_sem_ataque3_rotulo, dia_sem_ataque4_rotulo))






#dias com ataques.
# dia_1_x, dia_1_y,\
# dia_1_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/testing/testing_day_1.csv')


# dia_2_x, dia_2_y,\
# dia_2_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/testing/testing_day_2.csv')


# dia_3_x, dia_3_y,\
# dia_3_rotulo = ImportTrafficData.in_regression_model_20_sec_ugr16_format('../../../../../Data/ugr16/testing/testing_day_3.csv')



#
## Passo 2, Criar/importar um modelo de regressão do tensorflow
#
# Se o usuario executar o programa passando o argumento -t vamos configurar e treinar (criar) o modelo.
if sys.argv.count("-t"):

    # Passo 2.1, configurando o modelo:
    model = Sequential()
    model.add(Dense(16, input_dim=120, activation='sigmoid'))
    # A função de ativação vazia representa a função de ativação linear: y = x
    model.add(Dense(6, activation=None))
    # A função de perda também deve ser apropriada para modelos de regressão
    model.compile(loss='mse', optimizer='adam', metrics=['mse'])

    # Passo 2.2, treinando o modelo:
    # O modelo vai aprender a prever
    model.fit(
        x=dia_sem_ataque_x,
        y=dia_sem_ataque_y,
        validation_data=(dia_sem_ataque3_x, dia_sem_ataque3_y),
        batch_size=60,
        epochs=15
    )

    model.save('./model')

# Se nao, vamos importar um modelo ja previamente configurado e treinado.
else:
    model = load_model('model')

print("\nOrganização do modelo:")
model.summary()





previsao = model.predict(dia_sem_ataque3_x) # shape: (86380, 6)

plot_prediction_real_graph("Prediction-real graph:\npure_training_day_5 - MLP with a 6-neuron output layer", 
                           "./pure_training_day_5_1mlp6nol_prediction_real_graph(trained-with-3-4).jpg",
                           previsao, dia_sem_ataque3_y)






#
## Passo 3, encontrar o threshold que garanta o melhor f1 score para o modelo:
#

# 3.1, Colocando o modelo para prever um dia COM ataque. Já que o modelo
# só foi treinado para prever segundos sem ataque, ele provavelmente
# vai errar a previsão quando encontrar ataques. Em outras palavras, considerando
# a presença de anomalias, a diferença entre os atributos do segundo real e os
# do segundo previsto vai ser consideravelmente grande.
# Obs: o modelo errara a previsao tanto se ocorrer anomalias no segundo atual quanto no proximo.
#previsao = model.predict(dia_2_x) # shape: (86380, 6)


#
# 3.2, Cacular a diferença entre o que foi previsto pelo modelo e o valor real observado.
#      Espera-se que o erro seja maior nos segundos em que houve algum ataque.

# erro_abs_bytes_count   = abs(previsao[:, 0] - dia_2_y[:, 0])
# erro_abs_dst_ip_ent    = abs(previsao[:, 1] - dia_2_y[:, 1])
# erro_abs_dst_port_ent  = abs(previsao[:, 2] - dia_2_y[:, 2])
# erro_abs_src_ip_ent    = abs(previsao[:, 3] - dia_2_y[:, 3])
# erro_abs_src_port_ent  = abs(previsao[:, 4] - dia_2_y[:, 4])
# erro_abs_packets_count = abs(previsao[:, 5] - dia_2_y[:, 5])

# #erro_abs_geral: vetor de 86380 posicoes, onde cada posicao representa
# # o erro absoluto da previsao do modelo para aquele segundo.
# erro_abs_geral = erro_abs_dst_ip_ent + erro_abs_dst_port_ent + \
#                  erro_abs_src_ip_ent + erro_abs_src_port_ent + \
#                  erro_abs_bytes_count + erro_abs_packets_count  # shape: (86395, 1)


# 3.3, encontrar um numero (threshold) que representa o maximo que o valor real pode se diferenciar
#      do valor previsto para que o segundo ainda seja considerado como normal. Os segundos cujos
#      valores reais estejam fora desse threshold são considerados anomalos.

# melhor_f1 = -1
# melhor_threshold = -1

# # O thresold ideal sera aquele garanta o maior numero de acertos para as previsoes do modelo (melhor f1 score).
# I é o intervalo no qual vamos procurar o threshold ideal.
# I = np.linspace(2.5, 3.0, num=1000)
# for threshold in I:

#     # Ans é um array de 86380 posicoes que representa a resposta do modelo.
#     # obs: lembrar que o modelo começa a prever a partir do vigesimo segundo do dia por isso são apenas 86380 posicoes.
#     # Se ans[i] = 0, segundo o modelo, o i-esimo segundo é normal.
#     # Se ans[i] = 1, segundo o modelo, o i-esimo segundo é anomalo.
#     ans = np.zeros_like(erro_abs_geral)

#     # Os segundo nos quais o erro é maior que o threshold sao anomalos:
#     ans[erro_abs_geral > threshold] = 1

#     # Calcula o f1 score considerando as respostas atuais do modelo e as respostas corretas:
#     f1 = get_f1(ans, dia_2_rotulo)
#     #print("Threshold: {} - f1: {}".format(threshold, f1))

#     # Se esse f1 score for melhor que o melhor f1 ja calculado vamos salvar o threshold atual e esse novo f1.
#     if f1 > melhor_f1:
#         melhor_f1 = f1
#         melhor_threshold = threshold


# print('\n\nThreshold: ', melhor_threshold)

# #
# ## Passo 4, Testar o modelo em um novo dia COM anomalias usando o threashold encontrado:
# #
#
# previsao = model.predict(dia_3_x) # shape: (86380, 6)
#
# erro_abs_bytes_count   = abs(previsao[:, 0] - dia_3_y[:, 0])
# erro_abs_dst_ip_ent    = abs(previsao[:, 1] - dia_3_y[:, 1])
# erro_abs_dst_port_ent  = abs(previsao[:, 2] - dia_3_y[:, 2])
# erro_abs_src_ip_ent    = abs(previsao[:, 3] - dia_3_y[:, 3])
# erro_abs_src_port_ent  = abs(previsao[:, 4] - dia_3_y[:, 4])
# erro_abs_packets_count = abs(previsao[:, 5] - dia_3_y[:, 5])
#
# erro_abs_geral = erro_abs_dst_ip_ent + erro_abs_dst_port_ent + \
#                  erro_abs_src_ip_ent + erro_abs_src_port_ent + \
#                  erro_abs_bytes_count + erro_abs_packets_count  # shape: (86399, 1)
#
# ans = np.zeros_like(erro_abs_geral)
# ans[erro_abs_geral > melhor_threshold] = 1
#
# precision, recall, f1 = get_f1(ans, dia_3_rotulo, all_metrics=True)
# print('\n\nDay 3 Metrics:')
# print("Confusion matrix: ")
# tn, fp, fn, tp = confusion_matrix(dia_3_rotulo, ans).ravel()
# print(np.array([[tp, fp], [fn, tn]]))
# print('Precision: ', precision)
# print('Recall: ', recall)
# print('F1: ', f1)



# previsao = model.predict(dia_1_x) # shape: (86395, 6)
#
# erro_abs_bytes_count   = abs(previsao[:, 0] - dia_1_y[:, 0])
# erro_abs_dst_ip_ent    = abs(previsao[:, 1] - dia_1_y[:, 1])
# erro_abs_dst_port_ent  = abs(previsao[:, 2] - dia_1_y[:, 2])
# erro_abs_src_ip_ent    = abs(previsao[:, 3] - dia_1_y[:, 3])
# erro_abs_src_port_ent  = abs(previsao[:, 4] - dia_1_y[:, 4])
# erro_abs_packets_count = abs(previsao[:, 5] - dia_1_y[:, 5])
#
# erro_abs_geral = erro_abs_dst_ip_ent + erro_abs_dst_port_ent + \
#                  erro_abs_src_ip_ent + erro_abs_src_port_ent + \
#                  erro_abs_bytes_count + erro_abs_packets_count  # shape: (86399, 1)
#
# ans = np.zeros_like(erro_abs_geral)
# ans[erro_abs_geral > melhor_threshold] = 1
#
# precision, recall, f1 = get_f1(ans, dia_1_rotulo, all_metrics=True)
# print('\n\nDay 1 Metrics:')
# print("Confusion matrix: ")
# tn, fp, fn, tp = confusion_matrix(dia_1_rotulo, ans).ravel()
# print(np.array([[tp, fp], [fn, tn]]))
# print('Precision: ', precision)
# print('Recall: ', recall)
# print('F1: ', f1)
