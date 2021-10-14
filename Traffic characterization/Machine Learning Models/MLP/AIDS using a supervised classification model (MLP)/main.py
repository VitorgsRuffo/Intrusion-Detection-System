
#
# Modelo supervisionado de classificacao que implementa a parte principal de um Anomaly-based Intrusion Detection System (AIDS).
#
# Os data de entrada desse IDS são os atributos dos fluxos IP da rede, coletados ao longo de 1 segundo. Por exemplo,
# a soma de bytes e pacotes de todos os fluxos ip e a entropia do ip de origem de todos os fluxos observados em
# um certo segundo. O conjunto de dados, portanto, sera os atributos dos fluxos IP da rede, coletados segundo a segundo
# ao longo de um certo intervalo de tempo (geralmente um dia inteiro).
#
# Ele inicialmente aprendera o perfil de comportamento normal do trafego da rede (em um dia sem ataques).
# Com o perfil em maos, pode-se usar o modelo para analisar, segundo a segundo, o trafego da rede e determinar
# se existe evidencia de uma anomalia ou nao.
#
#OBS: this model is not working yet... It is predicting everything as normal.

import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
import keras.metrics as metrics

import sys
sys.path.insert(1, '../')
from ImportTrafficData import ImportTrafficData


#
## Passo 1, importar e preprocessar os dados :
#
# dia_n_x (input):
#          matriz com 86400 linhas e 6 colunas. Cada linha representa o fluxo de rede em um segundo do dia.
#          Cada coluna representa um atributo do fluxo de rede (referente ao segundo da linha).
#          Esses atributos são: numero de bytes, entropia de ip e porta de origem/destino, e numero de pacotes.
#
# dia_n_y (classificacao do input correspondente):
#           vetor de 86400 posicoes representando cada segundo do dia. Se o valor da posicao for 0
#           nao houve ataque naquele segundo, se for 1 houve ataque. No caso do dia 1, todas as labels serao 0, pois,
#           nesse dia não houve ataques.
dia_1_x, dia_1_y = ImportTrafficData.in_classification_model_format('../../../Data/orion/051218_60h6sw_c1_ht5_it0_V2_csv',
                               #Não ocorrem ataques nesse dia
                               [])


dia_2_x, dia_2_y = ImportTrafficData.in_classification_model_format('../../../Data/orion/051218_60h6sw_c1_ht5_it0_V2_csv_ddos_portscan',
                               # Os ataques acontecem nesses horários:
                               [('10:15:00', '11:30:00'),
                                ('13:25:00', '14:35:00')])



# dados sem evidencia de ataque que serao usados para a validacao durante o treinamento.
i = ImportTrafficData.time_to_seconds("15:00:00")
validacao_x = dia_2_x[i:]
validacao_y = dia_2_y[i:]


dia_3_x, dia_3_y = ImportTrafficData.in_classification_model_format('../../../Data/orion/171218_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos',
                               [('09:45:00', '11:10:00'),
                                ('17:37:00', '18:55:00')])

dia_4_x, dia_4_y = ImportTrafficData.in_classification_model_format('../../../Data/orion/120319_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos',
                               [('09:29:00', '10:44:00'),
                                ('16:16:00', '17:50:00')])


# Se o usuario executar o programa passando o argumento -t vamos configurar e treinar o modelo.
if sys.argv.count("-t"):
    #
    ## Passo 2, Configurar um modelo de previsao:
    #

    # Nesse modelo temos uma camada de entrada com 6 neuronios,
    # uma camada escondida com 10 neuronios e uma camada de saida com 1 neuronio.
    # A funcao de ativacao da camada interna e de saida é a sigmoid.
    model = Sequential()
    model.add(Dense(10, input_dim=6, activation='sigmoid'))
    #model.add(Dropout(0.2))
    model.add(Dense(1, activation=None))

    # Antes de treinar o modelo especifica-se algumas configuracoes de treinamento:
    # Funcao de perda: recebe pesos e biases como parametro. Ela entao calcula o quao ruim o modelo se saira
    # em classificar os data de treinamento caso ele utilize esse parametros.
    #
    # Otimizador: algoritmo que treina o modelo usando o conjunto de data de treinamento e a funcao de perda.
    # Treinar um modelo é encontrar um conjunto de pesos e biases que minimizam a funcao de perda
    # para um conjunto de data de treinamento.
    #
    # Metricas: usadas durante o treinamento para monitorar os resultados.
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    #
    ## Passo 3, Treinar modelo:
    #
    # O treinamento acontece da seguinte forma:
    # Teremos x etapas de treinamento chamadas de "epochs", e, para cada etapa:
    #    O dataset de treinamento é embaralhado e divido em agrupamentos (baches) de "bach_size" elementos.
    #    Para cada batch:
    #       O otimizador configurará a funcao de perda considerando os data do batch atual.
    #       Então ele executa um algoritmo que calcula o minimo dessa funcao encontrando, assim, os pesos/biases
    #       apropriados para o modelo aprender a classificar os elementos do batch atual.
    #    No fim de cada etapa, caso um dataset de validacao tenha sido especificado, é calculada a funcao de perda
    #    considerando os pesos/biases atuais e os data de validacao. Além disso, as metricas tambem sao calculadas.
    #    Isso nos ajuda a monitorar o progresso do aprendizado do modelo a cada epoch.

    model.fit(dia_1_x, dia_1_y,
              validation_data=(validacao_x, validacao_y),
              batch_size=60,
              epochs=10)

    model.save('./model')

# Se nao, vamos carregar um modelo ja previamente configurado e treinado.
else:
    model = load_model('model')

print("\nOrganização do modelo:")
model.summary()

#
## Passo 4,
#
#
#
## Agora podemos utilizar o modelo treinado para fazer previsoes sobre a ocorrencia ou nao de anomalias
#  em uma faixa de tempo de um dia qualquer utilizando os atributos do fluxo de rede referentes a cada
#  segundo dessa faixa:
#
#
print("\nPrevendo anomalias no dataset de teste 1...")
predictions = abs(model.predict(dia_3_x))
predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0

print("Confusion matrix: ")
tn, fp, fn, tp = confusion_matrix(dia_3_y, predictions).ravel()
print(np.array([[tp, fp], [fn, tn]]))


## Prevendo anomalias no dataset de teste:
#
print("\nPrevendo anomalias no dataset de teste 2...")
predictions = abs(model.predict(dia_4_x))
predictions[predictions >= 0.5] = 1
predictions[predictions < 0.5] = 0

print("Confusion matrix: ")
tn, fp, fn, tp = confusion_matrix(dia_4_y, predictions).ravel()
print(np.array([[tp, fp], [fn, tn]]))
