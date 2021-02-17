import numpy as np
import os

def get_small_utk_faces():
    X = np.load(os.path.join(os.path.dirname(__file__), "data/faces.npy"))
    # Augment data by left/right mirroring 
    X = np.concatenate([X, X[:, :, ::-1]], axis=0) 
    return X
