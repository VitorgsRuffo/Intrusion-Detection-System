from numpy.core.numeric import zeros_like
import pandas as pd
import numpy as np

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout


from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

import os
from pickle import dump, load


# Transforma uma string do formato
# 'hh:mm:ss' em quantidade de segundos à partir de 00:00:00
def get_sec(time:str):
    h, m, s = time.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def get_f1(ans:np.ndarray, labels:np.ndarray, all_metrics=False):
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
        precision = true_positive/(true_positive + false_positive)
    except ZeroDivisionError: # Caso não exista uma amostra dessa classe
        precision = 1.0

    try:
        recall = true_positive/(true_positive + false_negative)
    except ZeroDivisionError: # Caso não exista uma amostra dessa classe
        recall = 1.0
        
    f1 = 2 * ( (precision * recall) / (precision + recall) )

    if all_metrics:
        return precision, recall, f1

    return f1


def import_data(path: str, attack_intervals: list):
    # Lê os arquivos dos respectivos txt e os
    # formata para que fiquem no formato de matriz coluna
    bytes_file = np.loadtxt(os.path.join(path, 'bytes1.txt')).reshape(-1, 1)
    ip_dst_file = np.loadtxt(os.path.join(path, 'EntropiaDstIP1.txt')).reshape(-1, 1)
    port_dst_file = np.loadtxt(os.path.join(path, 'EntropiaDstPort1.txt')).reshape(-1, 1)
    ip_scr_file = np.loadtxt(os.path.join(path, 'EntropiaScrIP1.txt')).reshape(-1, 1)
    port_scr_file = np.loadtxt(os.path.join(path, 'EntropiaSrcPort1.txt')).reshape(-1, 1)
    packets_file = np.loadtxt(os.path.join(path, 'packets1.txt')).reshape(-1, 1)

    # Já guarda os valores brutos (que devem ser previstos) no Y
    data_y = np.column_stack((bytes_file, ip_dst_file, port_dst_file, ip_scr_file, port_scr_file, packets_file))

    # O scaler irá normalizar os dados, tornando-os mais
    # fáceis de trabalhar na rede neural.

    # Os dados devem ser normalizados com o mesmo scaler, então
    # um arquivo será salvo.

    # Para esse teste, o scaler será criado usando os dados de treinamento
    # já que estamos importando eles primeiro.
    #
    # Obs: O scaler agora será configurado em um dia sem ataque,
    # exatamente para não criar nenhum possível viés.

    try:
        _file = open('scalers.pkl', 'rb')
        scalers = load(_file)

    # Se o arquivo não existe, vamos criá-lo
    except IOError:

        # O normalizador precisa conhecer os dados antes de conseguir normalizá-los
        bytes_scaler = StandardScaler().fit(bytes_file)
        ip_dst_scaler = StandardScaler().fit(ip_dst_file)
        port_dst_scaler = StandardScaler().fit(port_dst_file)
        ip_scr_scaler = StandardScaler().fit(ip_scr_file)
        port_scr_scaler = StandardScaler().fit(port_scr_file)
        packets_scaler = StandardScaler().fit(packets_file)

        scalers = [bytes_scaler, ip_dst_scaler, port_dst_scaler, ip_scr_scaler, port_scr_scaler, packets_scaler]
        _file = open('scalers.pkl', 'wb')
        dump(scalers, _file)
        _file.close()

    # O comando transform faz a normalização
    bytes_file = scalers[0].transform(bytes_file)
    ip_dst_file = scalers[1].transform(ip_dst_file)
    port_dst_file = scalers[2].transform(port_dst_file)
    ip_scr_file = scalers[3].transform(ip_scr_file)
    port_scr_file = scalers[4].transform(port_scr_file)
    packets_file = scalers[5].transform(packets_file)

    # Juntando tudo em uma matriz só
    data_x = np.column_stack((bytes_file, ip_dst_file, port_dst_file, ip_scr_file, port_scr_file, packets_file))

    #
    ## Criar os rótulos
    #

    # Iniciando o array de rotulos com zeros (sem ataque nenhum)
    rotulos = np.zeros(data_x.shape[0])

    # attack_index guardará tuplas que representam os intervalos de ataque (em segundos).
    attack_index = []

    # Para cada intervalo de ataque, vamos converter para segundos
    for time_interval in attack_intervals:
        start = get_sec(time_interval[0])
        finish = get_sec(time_interval[1])
        attack_index.append((start, finish))

    #  Adicionar ataques em y:

    # Para cada intervalo de ataque...
    for time_interval in attack_index:

        # Os rotulos que estao dentro deste intervalo devem ser 1.
        for i in range(time_interval[0], time_interval[1]):
            rotulos[i] = 1

    # Removendo o ultimo elemento do data_x e o primeiro elemento do data_y.
    # Desta forma vamos treinar a rede para computar a seguinte funcao f(data_x[i]) = data_y[i+1],
    # ou seja, vamos utilizar os dados normalizados de um segundo arbitrario i para prever os dados
    # brutos do proximo segundo (i+1).
    data_x = data_x[:-1]
    data_y = data_y[1:]
    rotulos = rotulos[1:]

    return data_x, data_y, rotulos


#
## Passo 1, importar os dados
#
try:
    os.remove('scalers.pkl') # Deleta scaler pra não correr o risco de misturar com outros códigos
except FileNotFoundError:
    pass

# O scaler agora será configurado em um dia sem ataque,
#  exatamente para não criar nenhum possível viés


# Lembrando que agora o Y é o valor real do dado, que deve ser previsto pela rede neural
# O rótulo indica se o fluxo é anômalo ou não

dia_sem_ataque_x, dia_sem_ataque_y,\
dia_sem_ataque_rotulo = import_data('../../../data/051218_60h6sw_c1_ht5_it0_V2_csv', [])


dia_1_x, dia_1_y,\
dia_1_rotulo = import_data('../../../data/051218_60h6sw_c1_ht5_it0_V2_csv_ddos_portscan',
                            # Os ataques acontecem nesses horários:
                             [('10:15:00', '11:30:00'),
                            ('13:25:00', '14:35:00')])

dia_2_x, dia_2_y,\
dia_2_rotulo = import_data('../../../data/171218_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos',
                            [('09:45:00', '11:10:00'),
                            ('17:37:00', '18:55:00')])

dia_3_x, dia_3_y,\
dia_3_rotulo = import_data('../../../data/120319_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos',
                        [('09:29:00', '10:44:00'),
                        ('16:16:00', '17:50:00')])



# Vamos filtrar os segundos sem ataque do dia 1 para usar como conjunto de validacao durante o treinamento
dia_1_x = dia_1_x[dia_1_rotulo == 0, :]  #accessing elements in numpy ndarrays: [start:end:step, start:end:step, ...]
dia_1_y = dia_1_y[dia_1_rotulo == 0, :]
dia_1_rotulo = dia_1_rotulo[dia_1_rotulo == 0]


#
## Passo 2, Criar um modelo do tensorflow
#


model = Sequential()
model.add(Dense(16, input_dim=6, activation='sigmoid'))

# A função de ativação vazia representa a função de ativação linear: y = x
model.add(Dense(1, activation=None)) 

# A função de perda também deve ser apropriada para modelos de regressão
model.compile(loss='mse', optimizer='adam', metrics=['mse'])


#
## Passo 3, Treinar modelo
#

# No meu tcc, eu fiz um modelo que só previa a entropia de porta de destino, que está na posição 2

model.fit(
    x=dia_sem_ataque_x,
    y=dia_sem_ataque_y[:, 2],
    validation_data=(dia_1_x, dia_1_y[:, 2]),
    batch_size=60,
    epochs=20
)

#
## Passo 4, Encontrar limiar ideal
#
previsao = model.predict(dia_2_x) # shape: (86400, 1)
previsao = previsao.reshape(-1)   # shape: (86400,)

erro_abs = abs(previsao - dia_2_y[:, 2])

melhor_f1 = -1
melhor_limiar = -1

# I é o intervalo em que vamos procurar o erro
I = np.arange(0.01, 1, 0.01)
for limiar in I:
    # Ans é a resposta do modelo
    ans = np.zeros_like(erro_abs)

    # Os lugares em que o erro é maior que o limiar, são anomalias:
    ans[erro_abs > limiar] = 1

    f1 = get_f1(ans, dia_2_rotulo)

    # Atualizar melhor limiar
    if f1 > melhor_f1:
        melhor_f1 = f1
        melhor_limiar = limiar


print('\n\nMelhor Limiar: ', melhor_limiar)




#
## Passo 4, Testar
#

previsao = model.predict(dia_3_x)
previsao = previsao.reshape(-1) # shape: (86400,)

erro_abs = abs(previsao - dia_3_y[:, 2])

ans = np.zeros_like(erro_abs)
ans[erro_abs > melhor_limiar] = 1

precision, recall, f1 = get_f1(ans, dia_3_rotulo, all_metrics=True)

print('\nPrecisão (Normal): ', recall)
print('Revocação (Anomaly): ', precision)
print('F1: ', f1)