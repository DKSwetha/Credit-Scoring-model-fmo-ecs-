# data/data_utils.py
import numpy as np

def split_among_clients(X, y, n_clients=3, random_state=0):
    rng = np.random.RandomState(random_state)
    idx = np.arange(len(X))
    rng.shuffle(idx)
    splits = np.array_split(idx, n_clients)
    clients_data = []
    for s in splits:
        clients_data.append((X.iloc[s].reset_index(drop=True), y.iloc[s].reset_index(drop=True)))
    return clients_data
