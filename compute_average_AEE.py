from pathlib import Path
import numpy as np


metrics_file = 'saved/eraft_attention_selayer_3/metrics_Attention_SELayer.txt'
file = np.genfromtxt(metrics_file, delimiter=',')
AEE = file[:, 0]
outlier = file[:, 4]
print(outlier)

mean_AEE = np.mean(AEE)
mean_outlier = np.mean(outlier)

print('The average AEE is {}, average outlier is {}'.format(mean_AEE, mean_outlier))


