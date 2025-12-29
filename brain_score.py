# brain score between 3 and 7

# coding=utf-8
import numpy as np
from brainscore_vision import load_metric
from brainscore_vision.metrics.regression_correlation import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainio.assemblies import NeuroidAssembly

# act_clu # (3,v_,720), brain
# act_ly_dr # (7,u_,720), model
# 只要是两个二维矩阵，都可以用这个函数进行计算

# cross regressed correlation
regression = pls_regression()
correlation = pearsonr_correlation()
metric = CrossRegressedCorrelation(regression, correlation)

def sco(data1, data2):
    x = data1 # (samp,feat)
    y = data2 # (samp,feat)
    assembly1 = NeuroidAssembly(x,       
                        coords={'stimulus_id': ('presentation', np.arange(x.shape[0])),
                               'object_name': ('presentation', ['a'] * x.shape[0]),
                               'neuroid_id': ('neuroid', np.arange(x.shape[1])),
                                'region': ('neuroid', [0] * x.shape[1])},
                        dims=['presentation', 'neuroid'])
    assembly2 = NeuroidAssembly(y,
                       coords={'stimulus_id': ('presentation', np.arange(y.shape[0])),
                               'object_name': ('presentation', ['a'] * y.shape[0]),
                               'neuroid_id': ('neuroid', np.arange(y.shape[1])),
                                'region': ('neuroid', [0] * y.shape[1])},
                       dims=['presentation', 'neuroid'])
    score = metric(source=assembly1, target=assembly2)
    return score

if __name__ == '__main__':
    score = []
    for i in range(3):
        for j in range(7):
            asse1 = act_ly_dr[j] # (720,u_)
            asse2 = act_clu[i].T # (720,v_)
            score_ = sco(asse1,asse2)
            score.append(score_.data)

    score = np.array(score).reshape(3,7)