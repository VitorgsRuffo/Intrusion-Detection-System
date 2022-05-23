import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

"""
    graph_name: nome (string) que sera printado no cabeçalho do grafico.
    save_path: caminho (string) para salvar o grafico.
    predicted_data: array 2D com os dados previstos (cada linha é um segundo e cada coluna é uma feature)
    real_data: array 2D com os dados reais (cada linha é um segundo e cada coluna é uma feature)
    attack_intervals: lista de tuplas, onde cada tupla representa um intervalo de ataque (e.g., [('09:45:00', '11:10:00'), ('17:37:00', '18:55:00')]).
"""
def plot_prediction_real_graph(graph_name: str, save_path: str, predicted_data, real_data, attack_intervals: list):
    features_names = ['bytes', 'dst_ip_entropy', 'dst_port_entropy',
                      'src_ip_entropy', 'src_port_entropy', 'packets']
    mpl.rcParams['lines.linewidth'] = 0.5
    #mpl.style.use('seaborn') #['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
    figure, plots = plt.subplots(3, 2, figsize=(10.80,7.20))
    plt.suptitle(graph_name, fontsize=14)

    lin_space = np.linspace(0, 24, predicted_data.shape[0])
    for k in range(0, 6):
        i = k // 2 # mapping array index to matrix indices. (obs: 2 == matrix line length)
        j = k % 2
        plots[i][j].set_title(features_names[k])
        plots[i][j].step(lin_space, real_data[:, k], label='REAL', color='green')
        plots[i][j].step(lin_space, predicted_data[:, k], label='PREDICTED', color='orange')
        plots[i][j].set_xticks(np.arange(0, 25, 2))
        plots[i][j].margins(x=0)
        plots[i][j].legend(fontsize='xx-small')

        
        for attack_interval in attack_intervals:

            start, end = lin_space[attack_interval[0]], lin_space[attack_interval[1]]
            plots[i][j].axvspan(start, end, label="ATTACK", color='r', alpha=0.5)

        if k == 0:
            #plots[i][j].set_ylim([0, x])
            plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        elif k == 5:
            #plots[i][j].set_ylim([0, y])
            plots[i][j].ticklabel_format(axis="y", style="sci", scilimits=(0,0))
        else:
            plots[i][j].set_ylim([-1, 2])

    plt.tight_layout()
    plt.savefig(save_path, format='jpg', dpi=800)
