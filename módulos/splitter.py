import pandas as pd
import numpy as np

class UnderSampleSplit:
    def __init__(
        self,
        train_size=0.7, train_prct=1,
        test_size=None, test_prct=None,
        replace=False, shuffle=True,
        random_state=0
    ):
        self.train_size=train_size; self.train_prct=train_prct
        self.test_size=test_size; self.test_prct=test_prct
        self.replace=replace; self.shuffle=shuffle
        self.random_state=random_state

    def set_params(
        self, train_size=0.8, train_prct=1,
        test_size=None, test_prct=None,
    ):
        self.train_size=train_size; self.train_prct=train_prct
        self.test_size=test_size; self.test_prct=test_prct
        
    def split(self, _Y, n_splits=5, param_list=None):
        cv = []
        for i in range(n_splits):
            rs = self.random_state+i if self.random_state is not None else None
            if param_list is not None:
                self.set_params(**param_list[i])
            cv.append(self.train_test_undersample(_Y, random_state=rs))
        return cv

    def train_test_undersample(self, _Y, random_state):
        train_index, test_index = self.undersample(
            _Y, self.train_size, self.train_prct,
            shuffle=self.shuffle, replace=self.replace,
            random_state=random_state
        )
        if self.test_size is not None:
            adj_test_size = self.test_size / (1 - self.train_size)
            test_index, _left_index = self.undersample(
                _Y.loc[test_index],
                train_size=adj_test_size, train_prct=self.test_prct,
                shuffle=self.shuffle, replace=False,
                random_state=random_state
            )
        return train_index, test_index
    
    def undersample(
        self, _Y,
        train_size=0.7, train_prct=1,
        shuffle=True, replace=False,
        random_state=0
    ):
        Y = _Y.copy()
        class_count = Y.value_counts()
        majo_size, mino_size = (class_count.loc[i] for i in (0, 1))
        if train_prct is None: train_prct = majo_size / mino_size
        n_mino = int(round(mino_size * train_size))
        n_majo = int(round(n_mino * train_prct))
        rng = np.random.default_rng(random_state)
        train_index = []
        for label, n_samples in zip((0, 1), (n_majo, n_mino)):
            train_index += list(rng.choice(
                Y.index[Y==label],
                n_samples, replace
            ))
        test_index = list(set(Y.index).difference(train_index))
        if shuffle: rng.shuffle(train_index); rng.shuffle(test_index)
        return np.array(train_index), np.array(test_index)