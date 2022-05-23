import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.under_sampling import NearMiss

def get_sec(self, time:str):
    h, m, s = time.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def scale_day(day:np.ndarray, scalers:list) -> np.ndarray:
    for i, scaler in enumerate(scalers):
        day[:, i] = scaler.transform(day[:, i])
    return day

def import_data_globecom(should_scale_data:bool=True)->tuple:
    # 0 = Normal
    # 1 = Portscan
    # 2 = DDoS
    
    #orion (first)
    #filename = r'../../../Data/orion/first/start-based/'
    #filename = r'../../../Data/orion/first/duration-based/'    
    #dia_sem_ataque = np.array(pd.read_csv(filename+'051218_preprocessed.csv'))
    #dia_1_com_ataque = np.array(pd.read_csv(filename+'120319_portscan_ddos_preprocessed.csv'))
    #dia_1_com_ataque = np.array(pd.read_csv(filename+'051218_portscan_preprocessed.csv'))
    #dia_2_com_ataque = np.array(pd.read_csv(filename+'171218_portscan_ddos_preprocessed.csv'))


    #orion (second)
    #filename = r'../../../Data/orion/second/start-based/'
    filename = r'../../../Data/orion/second/duration-based/'    
    dia_sem_ataque = np.array(pd.read_csv(filename+'071218_preprocessed.csv'))
    dia_1_com_ataque = np.array(pd.read_csv(filename+'071218_ddos_portscan_preprocessed.csv'))
    dia_2_com_ataque = np.array(pd.read_csv(filename+'071218_portscan_ddos_preprocessed.csv'))


    #deprecated...
    #filename = r'E:\google_drive_faculdade\TCC\tcc-anomaly-detection\globecom2022\dados/'
    # filename = r'../../../Data/orion/second/start-based/daniels/'
    # dia_sem_ataque = np.array(pd.read_csv(filename+'071218_140h6sw_c1_ht5_it0_V2_csv.csv'))
    # dia_1_com_ataque = np.array(pd.read_csv(filename+'071218_140h6sw_c1_ht5_it0_V2_csv_ddos_portscan.csv'))
    # dia_2_com_ataque = np.array(pd.read_csv(filename+'071218_140h6sw_c1_ht5_it0_V2_csv_portscan_ddos.csv'))


    if should_scale_data:
        # scaler = StandardScaler().fit(dia_sem_ataque[:, 0:6])
        scaler = MinMaxScaler((-1, 1)).fit(dia_sem_ataque[:, 0:6])
        dia_sem_ataque[:, 0:6] = scaler.transform(dia_sem_ataque[:, 0:6])
        dia_1_com_ataque[:, 0:6] = scaler.transform(dia_1_com_ataque[:, 0:6])
        dia_2_com_ataque[:, 0:6] = scaler.transform(dia_2_com_ataque[:, 0:6])



    return dia_sem_ataque, dia_1_com_ataque, dia_2_com_ataque


def import_data_ugr(should_scale_data:bool=True, should_downsample:bool=True)->tuple:
    # 0 = Normal
    # 1 = Portscan
    # 2 = DDoS
    filename = r'F:\UGR dataset/'

    dia_sem_ataque = np.array(pd.read_csv(filename+r'training/pure_training_day_3.csv'))
    dia_1_com_ataque = np.array(pd.read_csv(filename+r'testing/testing_day_1.csv'))
    dia_2_com_ataque = np.array(pd.read_csv(filename+r'testing/testing_day_2.csv'))


    if should_scale_data:
        # scaler = StandardScaler().fit(dia_sem_ataque[:, 0:6])
        scaler = MinMaxScaler((-1, 1)).fit(dia_sem_ataque[:, 0:6])
        dia_sem_ataque[:, 0:6] = scaler.transform(dia_sem_ataque[:, 0:6])
        dia_1_com_ataque[:, 0:6] = scaler.transform(dia_1_com_ataque[:, 0:6])
        dia_2_com_ataque[:, 0:6] = scaler.transform(dia_2_com_ataque[:, 0:6])

    if should_downsample:
        print('Iniciando undersample...')
        # Trocando rótulos todos para 1
        dia_1_com_ataque[dia_1_com_ataque[:, 6]==2, 6] = 1

        # Mostrando valores antes
        unique, counts = np.unique(dia_1_com_ataque[:, 6], return_counts=True)
        print('Antes: \n', dict(zip(unique, counts)))

        sampler = NearMiss(version=2)
        x, y = sampler.fit_resample(dia_1_com_ataque[:, :6], dia_1_com_ataque[:, 6])
        dia_1_com_ataque = np.column_stack([x, y])
        


        unique, counts = np.unique(dia_1_com_ataque[:, 6], return_counts=True)
        print('Depois:\n', dict(zip(unique, counts)))

        print('Undersample finalizado.')



    return dia_sem_ataque, dia_1_com_ataque, dia_2_com_ataque

def import_data_cic(should_scale_data:bool=True)->tuple:
    # 0 = Normal
    # 1 = Portscan
    # 2 = DDoS
    # folder_name = r'E:\google_drive_faculdade\TCC\tcc-anomaly-detection\cicddos_marcos'

    # folder_name = r'E:\google_drive_faculdade\TCC\tcc-anomaly-detection\cic_redownsample'
    folder_name = r'E:\google_drive_faculdade\TCC\tcc-anomaly-detection\backup_mitigation_cicddos\cic_redownsample'

    # # Dia 1
    # dia_sem_ataque = np.array(pd.read_csv(folder_name+'/dia_1_sem_ataque.csv'))
    # dia_1_com_ataque = np.array(pd.read_csv(folder_name+'/dia_1_com_ataque.csv'))
    # dia_1_rotulo = np.array(pd.read_csv(folder_name+'/dia_1_rotulos.csv'))

    # # Dia 2
    # dia_2_com_ataque = np.array(pd.read_csv(folder_name+'/dia_2_com_ataque.csv'))
    # dia_2_rotulo = np.array(pd.read_csv(folder_name+'/dia_2_rotulos.csv'))

    # Dia 1
    dia_sem_ataque = np.array(pd.read_csv(folder_name+r'/01-12-train_sem_ataque_s_segundos_vazios.csv'))[:, :6]
    dia_1_com_ataque = np.array(pd.read_csv(folder_name+r'/01-12-train_com_ataque_s_segundos_vazios.csv'))[:, :6]
    dia_1_rotulo = np.array(pd.read_csv(folder_name+r'/01-12-train_rotulos_s_segundos_vazios.csv'))[:, :6]

    # Dia 2
    dia_2_com_ataque = np.array(pd.read_csv(folder_name+r'/03-11-test_com_ataque_s_segundos_vazios.csv'))[:,:6]
    dia_2_rotulo = np.array(pd.read_csv(folder_name+r'/03-11-test_rotulos_s_segundos_vazios.csv'))[:,:6]

    dia_sem_ataque[dia_sem_ataque > 1E308] = 0
    dia_1_com_ataque[dia_1_com_ataque > 1E308] = 0
    dia_2_com_ataque[dia_2_com_ataque > 1E308] = 0


    # Adicionando os rótulos
    dia_sem_ataque = np.concatenate([dia_sem_ataque, np.zeros((dia_sem_ataque.shape[0], 1))], axis=1)
    dia_1_com_ataque = np.concatenate([dia_1_com_ataque, dia_1_rotulo], axis=1)
    dia_2_com_ataque = np.concatenate([dia_2_com_ataque, dia_2_rotulo], axis=1)


    if should_scale_data:
        # scaler = StandardScaler().fit(dia_sem_ataque[:, 0:6])
        scaler = MinMaxScaler((-1, 1)).fit(dia_sem_ataque[:, 0:6])
        dia_sem_ataque[:, 0:6] = scaler.transform(dia_sem_ataque[:, 0:6])
        dia_1_com_ataque[:, 0:6] = scaler.transform(dia_1_com_ataque[:, 0:6])
        dia_2_com_ataque[:, 0:6] = scaler.transform(dia_2_com_ataque[:, 0:6])



    return dia_sem_ataque, dia_1_com_ataque, dia_2_com_ataque

def get_sec(time:str):
    h, m, s = time.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)

def import_data_old(should_scale_data:bool=True):
    root = r'E:\google_drive_faculdade\TCC\tcc-anomaly-detection\dados/'
    paths=[r'051218_60h6sw_c1_ht5_it0_V2_csv',
           r'051218_60h6sw_c1_ht5_it0_V2_csv_ddos_portscan',
           r'171218_60h6sw_c1_ht5_it0_V2_csv_portscan_ddos']

    ataques = [[],
               [['10:15:00', '11:30:00'], ['13:25:00', '14:35:00']],
               [['09:45:00', '11:10:00'], ['17:37:00', '18:55:00']]]

    datasets = []
    for path, times in zip(paths, ataques):
        bytes_file      = np.loadtxt( root+path+r'/bytes1.txt')
        ip_dst_file     = np.loadtxt( root+path+r'/EntropiaDstIP1.txt')
        port_dst_file   = np.loadtxt( root+path+r'/EntropiaDstPort1.txt')
        ip_scr_file     = np.loadtxt( root+path+r'/EntropiaScrIP1.txt')
        port_scr_file   = np.loadtxt( root+path+r'/EntropiaSrcPort1.txt')
        packets_file    = np.loadtxt( root+path+r'/packets1.txt')
        
        rotulo = np.zeros_like(bytes_file)
        for start_end in times:
            rotulo[get_sec(start_end[0]):get_sec(start_end[1])] = 1
        
        dataset = np.column_stack((bytes_file, ip_dst_file, port_dst_file, ip_scr_file, port_scr_file, packets_file, rotulo))
        datasets.append(dataset)

    if should_scale_data:
        scaler = MinMaxScaler((-1, 1)).fit(datasets[0][:, 0:6])
        datasets[0][:, 0:6] = scaler.transform(datasets[0][:, 0:6])
        datasets[1][:, 0:6] = scaler.transform(datasets[1][:, 0:6])
        datasets[2][:, 0:6] = scaler.transform(datasets[2][:, 0:6])

    return datasets

    
    
        


def stack_seconds(dataset:np.ndarray, seconds_in_batch:int):
    # Parece errado mas juro que tá certo
    # Mapeia o arrays pra ser o conjunto de segundos e o dataset pra ser o segundo seguinte
    if seconds_in_batch > 1:
        arrays = []

        for i, j in zip(range(seconds_in_batch), range(seconds_in_batch-1, 0, -1)):
            arrays.append(dataset[i:-j])
        
        arrays.append(dataset[seconds_in_batch-1:])
    
        return dataset[seconds_in_batch:], np.stack(arrays, axis=1)[:-1]
    else:
        return dataset[seconds_in_batch:], dataset[:-seconds_in_batch]

def fix_data(input:np.ndarray, output:np.ndarray):
    return input[1:, :], output[:-1, :]
