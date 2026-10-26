"""Frozen submission121 policy; feature arithmetic comes from aij26_final c734d56.

Pickle globals are restricted to the exact NumPy/sklearn types in these
checksum-pinned artifacts; arbitrary pickle functions are never resolved.
"""
from hashlib import sha256
import io
from pathlib import Path
import pickle

import numpy as np
from numpy._core.multiarray import _reconstruct, scalar
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.tree import ExtraTreeRegressor
from sklearn.tree._tree import Tree
from threadpoolctl import threadpool_limits

from nc17_features import WithEncoderROIs, map_features, selected_handcrafted

BASE = np.float64(.675)
CANDIDATES = np.asarray([.5, .6, .65, .675])
ARTIFACT_SHA256 = {
    'model.pt': '40841b9c867d5c6f1ed8fcfacefb14787a377d58f770da86e67067f1d57a4cf1',
    'policy.pkl': '0bf2ba074b3609aefea2eba383ab512f5d635311315c76a79e3b08472023de25',
    'encoder_transform.pkl': 'fc865212594a3e98eb3e9b1465d75225782a422bebc636f4a7f99ebc949b9758',
}


class FrozenUnpickler(pickle.Unpickler):
    allowed = {
        ('sklearn.ensemble._forest', 'ExtraTreesRegressor'): ExtraTreesRegressor,
        ('sklearn.tree._classes', 'ExtraTreeRegressor'): ExtraTreeRegressor,
        ('sklearn.tree._tree', 'Tree'): Tree,
        ('sklearn.preprocessing._data', 'StandardScaler'): StandardScaler,
        ('sklearn.decomposition._pca', 'PCA'): PCA,
        ('numpy._core.multiarray', '_reconstruct'): _reconstruct,
        ('numpy._core.multiarray', 'scalar'): scalar,
        ('numpy', 'ndarray'): np.ndarray,
        ('numpy', 'dtype'): np.dtype,
    }

    def find_class(self, module, name):
        try:
            return self.allowed[module, name]
        except KeyError:
            raise pickle.UnpicklingError(f'Unsupported calibration type: {module}.{name}') from None


def choose_thresholds(model, features, base_areas, pixel_counts):
    gains = model.predict(features)
    choice = gains.argmax(1)
    thresholds = CANDIDATES[choice].copy()
    thresholds[gains[np.arange(len(choice)), choice] <= 1e-8] = BASE
    thresholds[base_areas < .01 * pixel_counts] = BASE
    return thresholds


class FrozenCalibration:
    def __init__(self, folder: Path, model):
        artifacts = {}
        for name, expected in ARTIFACT_SHA256.items():
            raw = (folder / name).read_bytes()
            if sha256(raw).hexdigest() != expected:
                raise ValueError(f'Unexpected submission121 artifact: {name}')
            if name.endswith('.pkl'):
                artifacts[name] = FrozenUnpickler(io.BytesIO(raw)).load()
        self.policy = artifacts['policy.pkl']
        self.transform = artifacts['encoder_transform.pkl']
        np.testing.assert_array_equal(self.policy['candidates'], CANDIDATES)
        if (self.policy['base'] != BASE or self.policy['model'].n_features_in_ != 276 or
                self.transform['scaler'].n_features_in_ != 3072 or self.transform['pca'].n_components_ != 32):
            raise ValueError('Unexpected frozen calibration geometry')
        self.policy['model'].n_jobs = 1
        self.used = {int(i) for tree in self.policy['model'].estimators_ for i in tree.tree_.feature if i >= 0}
        self.net = WithEncoderROIs(model).eval()

    def threshold(self, image, probability, encoder):
        base_area = int((probability >= BASE).sum())
        if base_area < .01 * probability.size:
            return BASE
        with threadpool_limits(limits=1):
            embedding = self.transform['pca'].transform(self.transform['scaler'].transform(encoder))
            features = np.concatenate([map_features(probability),
                                       *selected_handcrafted(image, probability, self.used), embedding[0]])[None]
            if features.shape != (1, 276) or not np.isfinite(features).all():
                raise ValueError('Invalid submission121 feature vector')
            return choose_thresholds(self.policy['model'], features,
                                     np.asarray([base_area]), np.asarray([probability.size]))[0]
