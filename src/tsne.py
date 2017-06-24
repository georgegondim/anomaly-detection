import pandas as pd
import numpy as np
import importlib.util
from sklearn.preprocessing import StandardScaler
import time

spec = importlib.util.spec_from_file_location(
    "bhtsne", "/home/george/workspace/bhtsne/bhtsne.py")
bhtsne = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bhtsne)

df = pd.read_csv('../data/creditcard.csv')
print('Number of samples: %d' % df.shape[0])

df.drop('Time', 1)
df.drop('Class', 1)
standard_scaler = StandardScaler()
std_df = standard_scaler.fit_transform(df)
tic = time.time()

attr = bhtsne.run_bh_tsne(std_df, verbose=True, no_dims=2)
print("elapsed time to fit t-SNE: %.4f s" % (time.time() - tic))
np.save('tsne', attr)
