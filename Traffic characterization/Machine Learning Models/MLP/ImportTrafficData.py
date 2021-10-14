import numpy as np
import os
from pickle import dump, load
from sklearn.preprocessing import StandardScaler
import pandas as pd

"""
    Classe que define os metodos de importacao de dados do trafego de uma rede.
"""
class ImportTrafficData:
    """
    Converte uma string representando o formato de tempo 'hh:mm:ss'
    para a quantidade de segundos correspondente (à partir de 00:00:00).
    """
    @staticmethod
    def time_to_seconds(time: str):
        h, m, s = time.split(':')
        return int(h) * 3600 + int(m) * 60 + int(s)

    """
    Essa funcao importa os dados de fluxos IP no formato adequado para ser usado por
    modelos de classificacao.
    
    path: caminho para os arquivos com os dados
    anomalous_intervals: tuplas que representam os intervalos anomalos (no formato hh:mm:ss).
    """
    @staticmethod
    def in_classification_model_format(path: str, anomalous_intervals: list):
        # Lendo os arquivos que contêm os atributos dos fluxos ips de um dia completo.
        # Obs: Eles estao separados segundo a segundo.

        bytes_count = np.loadtxt(os.path.join(path, 'bytes1.txt')).reshape(-1, 1)   # shape: (86400, 1)
        dst_ip_ent = np.loadtxt(os.path.join(path, 'EntropiaDstIP1.txt')).reshape(-1, 1)
        dst_port_ent = np.loadtxt(os.path.join(path, 'EntropiaDstPort1.txt')).reshape(-1, 1)
        src_ip_ent = np.loadtxt(os.path.join(path, 'EntropiaScrIP1.txt')).reshape(-1, 1)
        src_port_ent = np.loadtxt(os.path.join(path, 'EntropiaSrcPort1.txt')).reshape(-1, 1)
        packets_count = np.loadtxt(os.path.join(path, 'packets1.txt')).reshape(-1, 1)

        # Um Scaler serve para normalizar dados, tornando mais facil o aprendizado de uma rede neural
        # que tem esses dados como entrada.
        # Ele sera criado considerando as caracteristicas do primeiro conjunto de dados a ser importado, e,
        # todos os diferentes conjuntos de dados devem ser normalizados com o mesmo scaler. Então, apos importar
        # o primeiro conjunto de dados (conjunto de treinamento) o scaler sera salvo em um arquivo para ser usado
        # para normalizar futuros conjuntos de dados (validacao/teste).

        try:
            _file = open('scalers.pkl', 'rb')
            scalers = load(_file)

        # Se o arquivo não existe, vamos criá-lo
        except IOError:

            # Os scalers sao criados levando em consideracao as caracteristicas dos dados a serem normalizados.
            bytes_count_scaler   = StandardScaler().fit(bytes_count)
            dst_ip_ent_scaler    = StandardScaler().fit(dst_ip_ent)
            dst_port_ent_scaler  = StandardScaler().fit(dst_port_ent)
            src_ip_ent_scaler    = StandardScaler().fit(src_ip_ent)
            src_port_ent_scaler  = StandardScaler().fit(src_port_ent)
            packets_count_scaler = StandardScaler().fit(packets_count)

            scalers = [bytes_count_scaler, dst_ip_ent_scaler, dst_port_ent_scaler,
                       src_ip_ent_scaler, src_port_ent_scaler, packets_count_scaler]
            _file = open('scalers.pkl', 'wb')
            dump(scalers, _file)
            _file.close()

        # Normalizando os dados
        bytes_count    = scalers[0].transform(bytes_count)
        dst_ip_ent     = scalers[1].transform(dst_ip_ent)
        dst_port_ent   = scalers[2].transform(dst_port_ent)
        src_ip_ent     = scalers[3].transform(src_ip_ent)
        src_port_ent   = scalers[4].transform(src_port_ent)
        packets_count  = scalers[5].transform(packets_count)

        # data_x (entrada do modelo de classificacao):
        #          matriz com 86400 linhas e 6 colunas. Cada linha representa o fluxo de rede em um segundo do dia.
        #          Cada coluna representa um atributo do fluxo de rede (referente ao segundo da linha).
        #          Esses atributos são: numero de bytes, entropia de ip e porta de destino/origem, e numero de pacotes.
        data_x = np.column_stack((bytes_count, dst_ip_ent, dst_port_ent, src_ip_ent, src_port_ent, packets_count))


        # data_y (classificacao da entrada correspondente (anomalo==1/normal==0):
        #           vetor de 86400 posicoes representando cada segundo do dia. Se o valor da posicao for 0
        #           nao houve uma anomalia naquele segundo, se for 1 houve anomalia.
        #
        data_y = np.zeros([86400])

        anomalous_intervals_in_seconds = []
        for anomalous_interval in anomalous_intervals:
            start = ImportTrafficData.time_to_seconds(anomalous_interval[0])
            finish = ImportTrafficData.time_to_seconds(anomalous_interval[1])
            anomalous_intervals_in_seconds.append((start, finish))

        # Para cada intervalo anomalo...
        for anomalous_interval in anomalous_intervals_in_seconds:
            # os segundos dentro desse intervalo sao marcados como anomalos ( rótulo == 1 ).
            for i in range(anomalous_interval[0], anomalous_interval[1]+1):
                data_y[i] = 1

        return data_x, data_y


    @staticmethod
    def in_classification_model_ugr16_format(path: str):
        # Lendo os arquivos que contêm os atributos dos fluxos ips de um dia completo.
        # Obs: Eles estao separados segundo a segundo.

        data_frame = pd.read_csv(path, dtype={"bytes": np.int32, "packets": np.int32, "label": np.int32})
        bytes_count = np.array(data_frame.bytes).reshape(-1, 1)
        src_ip_ent = np.array(data_frame.src_ip_entropy).reshape(-1, 1)
        dst_ip_ent = np.array(data_frame.dst_ip_entropy).reshape(-1, 1)
        src_port_ent = np.array(data_frame.src_port_entropy).reshape(-1, 1)
        dst_port_ent = np.array(data_frame.dst_port_entropy).reshape(-1, 1)
        packets_count = np.array(data_frame.packets).reshape(-1, 1)

        # Um Scaler serve para normalizar dados, tornando mais facil o aprendizado de uma rede neural
        # que tem esses dados como entrada.
        # Ele sera criado considerando as caracteristicas do primeiro conjunto de dados a ser importado, e,
        # todos os diferentes conjuntos de dados devem ser normalizados com o mesmo scaler. Então, apos importar
        # o primeiro conjunto de dados (conjunto de treinamento) o scaler sera salvo em um arquivo para ser usado
        # para normalizar futuros conjuntos de dados (validacao/teste).

        try:
            _file = open('scalers.pkl', 'rb')
            scalers = load(_file)

        # Se o arquivo não existe, vamos criá-lo
        except IOError:

            # Os scalers sao criados levando em consideracao as caracteristicas dos dados a serem normalizados.
            bytes_count_scaler = StandardScaler().fit(bytes_count)
            dst_ip_ent_scaler = StandardScaler().fit(dst_ip_ent)
            dst_port_ent_scaler = StandardScaler().fit(dst_port_ent)
            src_ip_ent_scaler = StandardScaler().fit(src_ip_ent)
            src_port_ent_scaler = StandardScaler().fit(src_port_ent)
            packets_count_scaler = StandardScaler().fit(packets_count)

            scalers = [bytes_count_scaler, dst_ip_ent_scaler, dst_port_ent_scaler,
                       src_ip_ent_scaler, src_port_ent_scaler, packets_count_scaler]
            _file = open('scalers.pkl', 'wb')
            dump(scalers, _file)
            _file.close()

        # Normalizando os dados
        bytes_count = scalers[0].transform(bytes_count)
        dst_ip_ent = scalers[1].transform(dst_ip_ent)
        dst_port_ent = scalers[2].transform(dst_port_ent)
        src_ip_ent = scalers[3].transform(src_ip_ent)
        src_port_ent = scalers[4].transform(src_port_ent)
        packets_count = scalers[5].transform(packets_count)

        # data_x (entrada do modelo de classificacao):
        #          matriz com 86400 linhas e 6 colunas. Cada linha representa o fluxo de rede em um segundo do dia.
        #          Cada coluna representa um atributo do fluxo de rede (referente ao segundo da linha).
        #          Esses atributos são: numero de bytes, entropia de ip e porta de destino/origem, e numero de pacotes.
        data_x = np.column_stack((bytes_count, dst_ip_ent, dst_port_ent, src_ip_ent, src_port_ent, packets_count))

        # data_y (classificacao da entrada correspondente (anomalo==1/normal==0):
        #           vetor de 86400 posicoes representando cada segundo do dia. Se o valor da posicao for 0
        #           nao houve uma anomalia naquele segundo, se for 1 houve anomalia.
        #
        data_y = np.array(data_frame.label)
        data_y[data_y == 2] = 1

        return data_x, data_y


    """
        Essa funcao importa os dados de fluxos IP no formato adequado para ser usado por
        modelos de regressao.
    
        path: caminho para os arquivos com os dados
        anomalous_intervals: tuplas que representam os intervalos anomalos (no formato hh:mm:ss).
    """
    @staticmethod
    def in_regression_model_format(path: str, anomalous_intervals: list):
        # Lendo os arquivos que contêm os atributos dos fluxos ips de um dia completo.
        # Obs: Eles estao separados segundo a segundo.

        bytes_count = np.loadtxt(os.path.join(path, 'bytes1.txt')).reshape(-1, 1)  # shape: (86400, 1)
        dst_ip_ent = np.loadtxt(os.path.join(path, 'EntropiaDstIP1.txt')).reshape(-1, 1)
        dst_port_ent = np.loadtxt(os.path.join(path, 'EntropiaDstPort1.txt')).reshape(-1, 1)
        src_ip_ent = np.loadtxt(os.path.join(path, 'EntropiaScrIP1.txt')).reshape(-1, 1)
        src_port_ent = np.loadtxt(os.path.join(path, 'EntropiaSrcPort1.txt')).reshape(-1, 1)
        packets_count = np.loadtxt(os.path.join(path, 'packets1.txt')).reshape(-1, 1)

        try:
            _file = open('scalers.pkl', 'rb')
            scalers = load(_file)

        except IOError:

            bytes_count_scaler = StandardScaler().fit(bytes_count)
            dst_ip_ent_scaler = StandardScaler().fit(dst_ip_ent)
            dst_port_ent_scaler = StandardScaler().fit(dst_port_ent)
            src_ip_ent_scaler = StandardScaler().fit(src_ip_ent)
            src_port_ent_scaler = StandardScaler().fit(src_port_ent)
            packets_count_scaler = StandardScaler().fit(packets_count)

            scalers = [bytes_count_scaler, dst_ip_ent_scaler, dst_port_ent_scaler,
                       src_ip_ent_scaler, src_port_ent_scaler, packets_count_scaler]

            _file = open('scalers.pkl', 'wb')
            dump(scalers, _file)
            _file.close()

        bytes_count = scalers[0].transform(bytes_count)
        dst_ip_ent = scalers[1].transform(dst_ip_ent)
        dst_port_ent = scalers[2].transform(dst_port_ent)
        src_ip_ent = scalers[3].transform(src_ip_ent)
        src_port_ent = scalers[4].transform(src_port_ent)
        packets_count = scalers[5].transform(packets_count)

        # data_x (entrada do modelo de rede neural):
        #          matriz com 86399 linhas e 6 colunas. Cada linha representa o fluxo de rede em um segundo do dia.
        #          Cada coluna representa um atributo do fluxo de rede (referente ao segundo da linha).
        #          Esses atributos são: numero de bytes, entropia de ip e porta de origem/destino, e numero de pacotes.
        #
        #          OBS: data_x representa os segundos do dia começando no 1.o segundo e indo ate o penultimo.
        #
        data_x = np.column_stack((bytes_count, dst_ip_ent, dst_port_ent,
                                  src_ip_ent, src_port_ent, packets_count))

        # data_y (saida que deve ser prevista pela rede neural):
        #          Semelhante a data_x.
        #
        #          OBS: data_y representa os segundos do dia começando no 2.o segundo e indo ate o ultimo.
        #
        #          Vamos treinar a rede para computar a seguinte funcao f(data_x[i]) = data_x[i+1],
        #          ou seja, vamos utilizar os atributos de um segundo arbitrario i para prever os atributos
        #          do proximo segundo (i+1).
        data_y = np.copy(data_x)

        # labels (classificacao de cada segundo do dia, i.e, 0==normal, 1==anomalo):
        #              vetor de 86399 posicoes representando cada segundo do dia (a partir do 2.o segundo do dia).
        #              Se o valor da posicao for 0 o trafego naquele segundo é normal, se for 1 ele é anomalo.
        #
        labels = np.zeros(86400)

        anomalous_intervals_in_seconds = []
        for anomalous_interval in anomalous_intervals:
            start = ImportTrafficData.time_to_seconds(anomalous_interval[0])
            finish = ImportTrafficData.time_to_seconds(anomalous_interval[1])
            anomalous_intervals_in_seconds.append((start, finish))

        # Para cada intervalo anomalo...
        for anomalous_interval in anomalous_intervals_in_seconds:
            # os segundos dentro desse intervalo sao marcados como anomalos ( rótulo == 1 ).
            for i in range(anomalous_interval[0], anomalous_interval[1] + 1):
                labels[i] = 1

        data_x = data_x[:-1]
        data_y = data_y[1:]
        labels = labels[1:]

        return data_x, data_y, labels

    """
            Essa funcao importa os dados de fluxos IP no formato adequado para ser usado por
            modelos de regressao.

            path: caminho para os arquivos com os dados
            anomalous_intervals: tuplas que representam os intervalos anomalos (no formato hh:mm:ss).
        """

    @staticmethod
    def in_regression_model_5_sec_format(path: str, anomalous_intervals: list):
        # Lendo os arquivos que contêm os atributos dos fluxos ips de um dia completo.
        # Obs: Eles estao separados segundo a segundo.

        bytes_count = np.loadtxt(os.path.join(path, 'bytes1.txt')).reshape(-1, 1)  # shape: (86400, 1)
        dst_ip_ent = np.loadtxt(os.path.join(path, 'EntropiaDstIP1.txt')).reshape(-1, 1)
        dst_port_ent = np.loadtxt(os.path.join(path, 'EntropiaDstPort1.txt')).reshape(-1, 1)
        src_ip_ent = np.loadtxt(os.path.join(path, 'EntropiaScrIP1.txt')).reshape(-1, 1)
        src_port_ent = np.loadtxt(os.path.join(path, 'EntropiaSrcPort1.txt')).reshape(-1, 1)
        packets_count = np.loadtxt(os.path.join(path, 'packets1.txt')).reshape(-1, 1)

        try:
            _file = open('scalers.pkl', 'rb')
            scalers = load(_file)

        except IOError:

            bytes_count_scaler = StandardScaler().fit(bytes_count)
            dst_ip_ent_scaler = StandardScaler().fit(dst_ip_ent)
            dst_port_ent_scaler = StandardScaler().fit(dst_port_ent)
            src_ip_ent_scaler = StandardScaler().fit(src_ip_ent)
            src_port_ent_scaler = StandardScaler().fit(src_port_ent)
            packets_count_scaler = StandardScaler().fit(packets_count)

            scalers = [bytes_count_scaler, dst_ip_ent_scaler, dst_port_ent_scaler,
                       src_ip_ent_scaler, src_port_ent_scaler, packets_count_scaler]

            _file = open('scalers.pkl', 'wb')
            dump(scalers, _file)
            _file.close()

        bytes_count = scalers[0].transform(bytes_count)
        dst_ip_ent = scalers[1].transform(dst_ip_ent)
        dst_port_ent = scalers[2].transform(dst_port_ent)
        src_ip_ent = scalers[3].transform(src_ip_ent)
        src_port_ent = scalers[4].transform(src_port_ent)
        packets_count = scalers[5].transform(packets_count)


        raw_data_x = np.column_stack((bytes_count, dst_ip_ent, dst_port_ent,
                                      src_ip_ent, src_port_ent, packets_count))


        data_y = np.copy(raw_data_x)[5:]
        data_x = np.zeros((data_y.shape[0], 30))

        for i in range(data_y.shape[0]):
            data_x[i] = raw_data_x[i:i+5].flatten()


        labels = np.zeros((86400, 1))

        anomalous_intervals_in_seconds = []
        for anomalous_interval in anomalous_intervals:
            start = ImportTrafficData.time_to_seconds(anomalous_interval[0])
            finish = ImportTrafficData.time_to_seconds(anomalous_interval[1])
            anomalous_intervals_in_seconds.append((start, finish))

        # Para cada intervalo anomalo...
        for anomalous_interval in anomalous_intervals_in_seconds:
            # os segundos dentro desse intervalo sao marcados como anomalos ( rótulo == 1 ).
            for i in range(anomalous_interval[0], anomalous_interval[1] + 1):
                labels[i] = 1

        labels = labels[5:]

        return data_x, data_y, labels


    @staticmethod
    def in_regression_model_20_sec_format(path: str, anomalous_intervals: list):
        # Lendo os arquivos que contêm os atributos dos fluxos ips de um dia completo.
        # Obs: Eles estao separados segundo a segundo.
        bytes_count = np.loadtxt(os.path.join(path, 'bytes1.txt')).reshape(-1, 1)  # shape: (86400
        dst_ip_ent = np.loadtxt(os.path.join(path, 'EntropiaDstIP1.txt')).reshape(-1, 1)
        dst_port_ent = np.loadtxt(os.path.join(path, 'EntropiaDstPort1.txt')).reshape(-1, 1)
        src_ip_ent = np.loadtxt(os.path.join(path, 'EntropiaScrIP1.txt')).reshape(-1, 1)
        src_port_ent = np.loadtxt(os.path.join(path, 'EntropiaSrcPort1.txt')).reshape(-1, 1)
        packets_count = np.loadtxt(os.path.join(path, 'packets1.txt')).reshape(-1, 1)

        try:
            _file = open('scalers.pkl', 'rb')
            scalers = load(_file)
        except IOError:
            bytes_count_scaler = StandardScaler().fit(bytes_count)
            dst_ip_ent_scaler = StandardScaler().fit(dst_ip_ent)
            dst_port_ent_scaler = StandardScaler().fit(dst_port_ent)
            src_ip_ent_scaler = StandardScaler().fit(src_ip_ent)
            src_port_ent_scaler = StandardScaler().fit(src_port_ent)
            packets_count_scaler = StandardScaler().fit(packets_count)
            scalers = [bytes_count_scaler, dst_ip_ent_scaler, dst_port_ent_scaler,
                       src_ip_ent_scaler, src_port_ent_scaler, packets_count_scaler]
            _file = open('scalers.pkl', 'wb')
            dump(scalers, _file)
            _file.close()

        bytes_count = scalers[0].transform(bytes_count)
        dst_ip_ent = scalers[1].transform(dst_ip_ent)
        dst_port_ent = scalers[2].transform(dst_port_ent)
        src_ip_ent = scalers[3].transform(src_ip_ent)
        src_port_ent = scalers[4].transform(src_port_ent)
        packets_count = scalers[5].transform(packets_count)

        raw_data_x = np.column_stack((bytes_count, dst_ip_ent, dst_port_ent,
                                      src_ip_ent, src_port_ent, packets_count))

        data_y = np.copy(raw_data_x)[20:]
        data_x = np.zeros((data_y.shape[0], 120))
        for i in range(data_y.shape[0]):
            data_x[i] = raw_data_x[i:i+20].flatten()

        labels = np.zeros((86400, 1))
        anomalous_intervals_in_seconds = []
        for anomalous_interval in anomalous_intervals:
            start = ImportTrafficData.time_to_seconds(anomalous_interval[0])
            finish = ImportTrafficData.time_to_seconds(anomalous_interval[1])
            anomalous_intervals_in_seconds.append((start, finish))
        # Para cada intervalo anomalo...
        for anomalous_interval in anomalous_intervals_in_seconds:
            # os segundos dentro desse intervalo sao marcados como anomalos ( rótulo == 1 ).
            for i in range(anomalous_interval[0], anomalous_interval[1] + 1):
                labels[i] = 1
        labels = labels[20:]
        return data_x, data_y, labels


    @staticmethod
    def in_regression_model_20_sec_ugr16_format(path: str):
        # Lendo os arquivos que contêm os atributos dos fluxos ips de um dia completo.
        # Obs: Eles estao separados segundo a segundo.
        data_frame = pd.read_csv(path, dtype={"bytes": np.int32, "packets": np.int32, "label": np.int32})

        bytes_count = np.array(data_frame.bytes).reshape(-1, 1)
        src_ip_ent = np.array(data_frame.src_ip_entropy).reshape(-1, 1)
        dst_ip_ent = np.array(data_frame.dst_ip_entropy).reshape(-1, 1)
        src_port_ent = np.array(data_frame.src_port_entropy).reshape(-1, 1)
        dst_port_ent = np.array(data_frame.dst_port_entropy).reshape(-1, 1)
        packets_count = np.array(data_frame.packets).reshape(-1, 1)

        try:
            _file = open('scalers.pkl', 'rb')
            scalers = load(_file)
        except IOError:
            bytes_count_scaler = StandardScaler().fit(bytes_count)
            dst_ip_ent_scaler = StandardScaler().fit(dst_ip_ent)
            dst_port_ent_scaler = StandardScaler().fit(dst_port_ent)
            src_ip_ent_scaler = StandardScaler().fit(src_ip_ent)
            src_port_ent_scaler = StandardScaler().fit(src_port_ent)
            packets_count_scaler = StandardScaler().fit(packets_count)
            scalers = [bytes_count_scaler, dst_ip_ent_scaler, dst_port_ent_scaler,
                       src_ip_ent_scaler, src_port_ent_scaler, packets_count_scaler]
            _file = open('scalers.pkl', 'wb')
            dump(scalers, _file)
            _file.close()

        bytes_count = scalers[0].transform(bytes_count)
        dst_ip_ent = scalers[1].transform(dst_ip_ent)
        dst_port_ent = scalers[2].transform(dst_port_ent)
        src_ip_ent = scalers[3].transform(src_ip_ent)
        src_port_ent = scalers[4].transform(src_port_ent)
        packets_count = scalers[5].transform(packets_count)

        raw_data_x = np.column_stack((bytes_count, dst_ip_ent, dst_port_ent,
                                      src_ip_ent, src_port_ent, packets_count))

        data_y = np.copy(raw_data_x)[20:]
        data_x = np.zeros((data_y.shape[0], 120))
        for i in range(data_y.shape[0]):
            data_x[i] = raw_data_x[i:i + 20].flatten()

        labels = np.array(data_frame.label)
        labels[labels == 2] = 1
        labels = labels[20:]

        return data_x, data_y, labels
