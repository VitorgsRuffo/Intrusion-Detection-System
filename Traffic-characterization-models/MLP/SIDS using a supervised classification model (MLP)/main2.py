
#
# Modelo supervisionado de classificacao que implementa a parte principal de um Signature-based Intrusion Detection System (SIDS).
#
# Os data de entrada desse IDS são os atributos dos fluxos ip da rede, coletados a cada 1 segundo. Por exemplo,
# a soma de bytes e pacotes de todos os fluxos ip e a entropia do ip de origem de todos os fluxos observados em
# um certo segundo.

# Ele aprendera os padroes (assinatura) de ataques conhecidos durante o seu treinamento. Esses padroes são mudanças
# anormais em atributos dos fluxos ip da rede que ocorrem quando os ataques acontecem. Esse tipo de IDS pode não ser
# capaz de reconhecer ataques cujo padrao não foi aprendido durante a fase de treinamento (alta taxa de falsos
# negativos). Porem, ele apresentara uma baixa taxa de falsos positivos ja que ele só soara um alarme quando um ataque
# conhecido acontecer.
#

import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
import keras.metrics as metrics
import os
import sys

from sklearn.model_selection import train_test_split


#sys.path.insert(1, '../../../data')
from ImportTrafficData import ImportTrafficData

try:
    os.remove('scalers.pkl') # Deleta scaler pra não correr o risco de misturar com outros códigos
except FileNotFoundError:
    pass

#
## Passo 1, importar e preprocessar os dados:
#
# dia_n_x (input):
#          matriz com 86400 linhas e 6 colunas. Cada linha representa o fluxo de rede em um segundo do dia.
#          Cada coluna representa um atributo do fluxo de rede (referente ao segundo da linha).
#          Esses atributos são: numero de bytes, entropia de ip e porta de origem/destino, e numero de pacotes.
#
# dia_n_y (classificacao do input correspondente):
#           vetor de 86400 posicoes representando cada segundo do dia. Se o valor da posicao for 0
#           nao houve ataque naquele segundo, se for 1 houve ataque.
data_x, data_y = ImportTrafficData.in_classification_model_format2('../')


#training_x, testing_x, training_y, testing_y = train_test_split(data_x, data_y, test_size=0.33)
#training_x, validating_x, training_y, validating_y = train_test_split(training_x, training_y, test_size=0.5)



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
    model.add(Dense(1, activation='sigmoid'))

    # Antes de treinar o modelo especifica-se algumas configuracoes de treinamento:
    # Funcao de perda: recebe pesos e biases como parametro. Ela entao calcula o quao ruim o modelo se saira
    # em classificar os dados de treinamento caso ele utilize esse parametros.
    #
    # Otimizador: algoritmo que treina o modelo usando o conjunto de data de treinamento e a funcao de perda.
    # Treinar um modelo é encontrar um conjunto de pesos e biases que minimizam a funcao de perda
    # para um conjunto de data de treinamento.
    #
    # Metricas: usadas durante o treinamento para monitorar os resultados.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy',
                                                                         metrics.Precision(),
                                                                         metrics.Recall()])

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

    model.fit(data_x, data_y,
              #validation_data=(validating_x, validating_y),
              batch_size=60,
              epochs=20)

    model.save('./model')

# Se nao, vamos carregar um modelo ja previamente configurado e treinado.
else:
    model = load_model('model')

print("\nOrganização do modelo:")
model.summary()

#
## Passo 4, Avaliando o modelo sobre o dataset de teste:
#
print("\nAvaliando o modelo sobre o dataset de teste:")
#model.evaluate(testing_x, testing_y, verbose=2)

#
#
## Agora podemos utilizar o modelo treinado para fazer previsoes sobre a ocorrencia ou nao de anomalias
#  em uma faixa de tempo de um dia qualquer utilizando os atributos do fluxo de rede referentes a cada
#  segundo dessa faixa:
#
#
## Prevendo anomalias no dataset de teste:
#
# print("\nPrevendo anomalias no dataset de teste...")
# predictions = model.predict(testing_x)
# predictions[predictions >= 1] = 2
# predictions[predictions < 1] = 0
#
# print("Confusion matrix: ")
# tn, fp, fn, tp = confusion_matrix(testing_y, predictions).ravel()
# print(np.array([[tp, fp], [fn, tn]]))


# figure, axis = plt.subplots(1, 2, constrained_layout=True)
# #plotando o grafico dos rotulos do conjunto de treinamento
# axis[0].set_title('Rotulos do conjunto de treinamento')
# axis[0].step(np.linspace(0, 86400, 86400), dia_3_y, color='green')
# #plotando o grafico das previsoes
# axis[1].set_title('Previsões sobre o conjunto de treinamento')
# axis[1].step(np.linspace(0, 86400, 86400), predictions, color='red')
#
# plt.axvspan(10000, 20000, color='r', alpha=0.5)
#
# plt.show()
#
#
#
# #
# ## Prevendo anomalias em um dia que não as possui...
# #
# dia_4_x, dia_4_y = import_data('../../data/051218_60h6sw_c1_ht5_it0_V2_csv', [])
#
# print("\nPrevendo anomalias em um dia que não as possui...")
# predictions = model.predict(dia_4_x)
# predictions[predictions >= 0.5] = 1
# predictions[predictions < 0.5] = 0
# print("Quantidade de segundos previstos erroneamente como anomalos:" + str(len(predictions[predictions == 1])))
