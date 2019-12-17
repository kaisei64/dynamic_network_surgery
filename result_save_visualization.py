import cloudpickle
import pandas as pd
import seaborn as sns
from draw_architecture import mydraw


def result_save(path, data_dict, input_data):
    for i, data in enumerate(data_dict):
        data_dict[data].append(input_data[i])
    df = pd.DataFrame.from_dict(data_dict)
    df.to_csv(path)


def parameter_save(path, param):
    with open(path, 'wb') as f:
        cloudpickle.dump(param, f)


def parameter_use(path):
    with open(path, 'rb') as f:
        return cloudpickle.load(f)


def parameter_distribution_vis(path, param):
    sns.set_style("darkgrid")
    sns_plot = sns.distplot(param, rug=True)
    sns_plot.figure.savefig(path)
    sns_plot.figure.clear()


# fl_numで取り出すチャネルを指定
# ch_numで取り出すチャネルを指定
def conv_vis(path, param, fl_num, ch_num=0):
    sns_map = sns.heatmap(param[fl_num, ch_num, :, :], vmin=0.0, vmax=1.0, annot=True)
    sns_map.figure.savefig(path)
    sns_map.figure.clear()


# 入力はパラメータのリスト
def dense_vis(param_list):
    mydraw(param_list)
