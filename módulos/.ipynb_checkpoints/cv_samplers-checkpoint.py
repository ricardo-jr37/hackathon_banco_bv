import pandas as pd
import numpy as np
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split as tts
from IPython.display import clear_output as co


def print_cls_cnt(Y, t_ind, e_ind):
    cls_cnt = pd.concat([Y[ind].value_counts().rename(col).to_frame() for ind, col in zip([t_ind, e_ind], ['Train set', 'Test set'])], 1)
    cls_cnt.index.name = 'Class'
    display(cls_cnt.fillna(''))
    
def print_intra_cls_cnt(y, groups, prct=False):
    inner_cnt = [y[groups==label].value_counts().rename(f'{label}') for label in groups.unique()]
    inner_total = [y.value_counts().rename('total - inner')]
    outter_total = [groups.value_counts().rename('total - outer')]
    counts = inner_cnt + inner_total + outter_total
    
    intra_cls_cnt = pd.concat(counts, 1)
    intra_cls_cnt.columns.name='outer class'; intra_cls_cnt.index.name='inner class (target)'
    
    if prct:
        intra_cls_cnt['0 (% - outer)'] = 100 * (intra_cls_cnt.loc[0, ['0', '1']].values / intra_cls_cnt['total - outer']).values
        intra_cls_cnt['1 (% - outer)'] = 100 * (intra_cls_cnt.loc[1, ['0', '1']].values / intra_cls_cnt['total - outer']).values
        intra_cls_cnt['0 (% - inner)'] = 100 * (intra_cls_cnt['0'].values / intra_cls_cnt['total - inner']).values
        intra_cls_cnt['1 (% - inner)'] = 100 * (intra_cls_cnt['1'].values / intra_cls_cnt['total - inner']).values        
         
    display(intra_cls_cnt.round(1).fillna(''))

def target_categorical_distribution(X, Y, col='ID_TURMA', verbose=0):
    minority_index = Y[Y==1].index
    turmas = sorted(list(X.loc[minority_index][col].unique()))
    turma_counts = {}
    for i, turma in enumerate(turmas):
        if (i%50==0 or (i+1)==len(turmas)) and verbose: co(wait=True); print(f'{col} evaluated: {i+1}/{len(turmas)}')
        turma_counts[turma] = Y[X[col]==turma].value_counts()
    return pd.DataFrame(turma_counts).fillna(0)[turmas]

def categorical_undersampling(X, Y, col='ID_TURMA', prct=None, cnt=None, random_state=0, shuffle=False, verbose=0):
    if cnt is None: cnt = target_categorical_distribution(X, Y, col, verbose)
    minority_msk = Y==1
    minority_index = Y[minority_msk].index
    turmas = cnt.columns.tolist()
    X_0 = X[minority_msk==False]
    XX = []
    for i, turma in enumerate(turmas):
        if verbose and ((i % 50) == 0 or i == (cnt.shape[1] - 1)):
            co(wait=True); print(f'Undersampling categories - {i+1}/{cnt.shape[1]} categories sampled.')
        if cnt[turma][0]:
            ctgr_0 =  X_0[X_0[col]==turma]
            majority_n_samples = round(cnt[turma][1] * prct)
            replace = majority_n_samples > len(ctgr_0)
            XX.append(ctgr_0.sample(majority_n_samples, replace=replace, random_state=random_state))
    XX = pd.concat(XX)
    x_ = pd.concat([XX, X[minority_msk]])
    if shuffle: x_ = x_.sample(len(x_), replace=False, random_state=random_state)
    y_ = Y.loc[x_.index].copy()
    test_index = set(Y.index).difference(y_.index)
    xe, ye = X.loc[test_index], Y.loc[test_index]
    return x_, y_, xe, ye

def GroupUnderSample(x, y, groups, prct=1, shuffle=True, replace=False, random_state=0, verbose=0):
    if type(groups)==str:
        groups = x[groups].copy()

    # minority_msk = y==1
    y_majo = y[y==0]
    y_mino = y[y==1]
    groups_majo = groups.loc[y_majo.index]
    groups_mino_count = groups.loc[y_mino.index].value_counts()

    unique_groups = np.sort(groups.unique())
    n_groups = len(unique_groups)
    index_majo = []
    for i, group in enumerate(unique_groups):
        if verbose and ((i % 50) == 0 or (i+1) == n_groups):
            co(wait=True); print(f'Undersampling categories independently - {i+1}/{n_groups} categories sampled.')

        y_group_majo = y_majo[groups_majo==group]
        mino_cnt = groups_mino_count[group]
        majo_n_samples = int(round(mino_cnt * prct))
        if not replace and majo_n_samples > y_group_majo.shape[0]: majo_n_samples = y_group_majo.shape[0]
        index_majo += y_group_majo.sample(majo_n_samples, replace=replace, random_state=random_state).index.tolist()

    train_index = np.array(index_majo + y_mino.index.tolist())
    test_index = np.array(list(set(y.index.tolist()).difference(train_index)))
    if shuffle:
        rand_gen = np.random.default_rng(random_state)
        rand_gen.shuffle(train_index);
        rand_gen.shuffle(test_index)

    return x.loc[train_index], x.loc[test_index], y.loc[train_index], y.loc[test_index]

def tts_categorical(x_, y_, test_size=0.2, col='ID_TURMA', cnt=None, random_state=0, verbose=0):
    if cnt is None: cnt = target_categorical_distribution(x_, y_, col, verbose)
    n_train_samples = round(y_.shape[0]*(1-test_size))
    cnt = cnt.T.sample(frac=1, random_state=random_state).T
    train_groups_msk = cnt.sum().cumsum() < n_train_samples
    train_groups = cnt.columns.values[train_groups_msk].tolist()
    train_msk = x_[col].isin(train_groups)
    xt = x_[train_msk].copy(); yt = y_.loc[train_msk].copy()
    xe = x_[train_msk==False].copy(); ye = y_.loc[train_msk==False].copy()
    return xt, xe, yt, ye

class GroupUnderSampleSplit:
    
    def __init__(
        self,
        n_splits=None,
        train_size=0.8, train_prct=1,             # related to minority class size in given sample   # related to minotiy class size in train and test sets
        test_size=None, test_prct=None,
        group_col=None, key_col=None,
        reset_index=False, remove_duplicates=True,
        random_state=0, verbose=1
    ):
        self.n_splits=n_splits
        self.train_size=train_size             # percentage of minority class size in train set
        self.train_prct=train_prct             # ratio of maiority/minority class size in train set
        self.test_size=test_size             # percentage of minority class size in train set
        self.test_prct=test_prct             # ratio of maiority/minority class size in train set
        self.group_col=group_col               # categorical column to use to perform group shuffle split
        self.key_col=key_col                   # categorical column to use to under sample per category. ignored for train or test samples if 'train_prct' or 'test_prct' is 'None'.
        self.reset_index=reset_index
        self.remove_duplicates=remove_duplicates
        self.random_state=random_state
        self.verbose=verbose
        self.base_train_size=train_size
        self.base_train_prct=test_prct
        self.adjusted_test_size = test_size / (1 - train_size) if train_size is not None and test_size is not None else None

    def set_params(
        self,
        train_size=0.8, train_prct=1,
        test_size=None, test_prct=None,
        group_col=None, key_col=None,
    ):
        self.train_size=train_size
        self.train_prct=train_prct
        self.test_size=test_size
        self.test_prct=test_prct
        self.group_col=group_col
        self.key_col=key_col
        self.base_train_size=train_size
        self.base_train_prct=test_prct
        self.adjusted_test_size = test_size / (1 - train_size) if train_size is not None and test_size is not None else None

    def undersample(self, _x, _y, cnt=None, shuffle=False, random_state=0):
        t_index, e_index = self.single_split_index(_x, _y, self.train_size, self.train_prct, cnt, shuffle, random_state)
        return _x.loc[t_index].copy(), _x.loc[e_index].copy(), _y.loc[t_index].copy(), _y.loc[e_index].copy()
          
    def train_test_split_index(self, _x, _y, cnt=None, shuffle=False, random_state=0):
        t_index, e_index_ = self.single_split_index(_x, _y, self.train_size, self.train_prct, cnt, shuffle, random_state)
        groups_is_list = self.key_col is not None and type(self.key_col)!=str
        if groups_is_list:
            original_groups = self.key_col.copy()
            self.key_col = self.key_col.loc[e_index_]
        e_index, e_index_ = self.single_split_index(_x.loc[e_index_], _y.loc[e_index_], self.adjusted_test_size, self.test_prct, cnt=None, shuffle=shuffle, random_state=random_state)
        if groups_is_list: self.key_col = original_groups
        
        if self.reset_index:
            reset_index = pd.Series(np.arange(_y.size), index=_y.index)
            t_index = reset_index[t_index]
            e_index = reset_index[e_index]
        return t_index, e_index

    def train_test_undersample(self, x, y, cnt=None, shuffle=False, random_state=0):
        _x, _y = x.copy(), y.copy()
        t_index, e_index = self.train_test_split_index(_x, _y, cnt, shuffle, random_state)
        if self.reset_index:
            _x.reset_index(drop=True, inplace=True)
            _y.reset_index(drop=True, inplace=True)
        return _x.loc[t_index].copy(), _x.loc[e_index].copy(), _y.loc[t_index].copy(), _y.loc[e_index].copy()
    
    def split(self, x, y, shuffle=True, params_list=None):
        cnt = None if self.key_col is None else target_categorical_distribution(x, y, col=self.key_col, verbose=1)
        train_indexes, test_indexes = [], []
        rs = self.random_state + 0
        for i in range(self.n_splits):
            co(wait=True); print(f'Performing splits - {i}/{self.n_splits} splits performed.')
            if rs is not None: rs+=i
            if params_list is not None: self.set_params(**params_list[i])
            train_index, test_index = self.train_test_split_index(x, y, cnt, shuffle, random_state=rs)
            train_indexes.append(train_index)
            test_indexes.append(test_index)
        co(wait=True); print(f'Performing splits - {i}/{self.n_splits} splits performed.')
        return list(zip(train_indexes, test_indexes))
    

    def single_split_index(
        self,
        _x, _y, train_size=0.2, train_prct=1, cnt=None, shuffle=False, random_state=0
    ):
        X, Y = _x.copy(), _y.copy()
#         invert = lambda label: int(not label)
        if self.key_col is None:
            sampling_strategy = 1 / train_prct
            rus = RandomUnderSampler(sampling_strategy, random_state=random_state)
            x, y = rus.fit_resample(X, Y)
            x.index = Y.index.values[rus.sample_indices_]
            y.index = x.index
        else:            
            x, y, _xe, _ye = categorical_undersampling(
                X, Y, col=self.key_col,
                prct=train_prct, cnt=cnt,
                random_state=random_state,
                shuffle=False, verbose=0
            )
        if self.group_col is None:
            stratify_strategy = y#(y if self.key_col is None else x[self.key_col].values)
            xt, xe_, yt, ye_ = tts(x, y, test_size=(1-train_size), random_state=random_state, shuffle=True, stratify=stratify_strategy)
        else:
            xt, xe_, yt, ye_ = tts_categorical(x, y, test_size=(1-train_size), col=self.group_col, cnt=None, random_state=random_state)
        train_index = np.array(yt.index) # duplicates could be removed here
        test_index = np.array(list(set(Y.index).difference(yt.index))) # includes _ye and ye_ indexes (left out in both samplings respectively)
        if shuffle:
            rand_gen = np.random.default_rng(random_state)
            rand_gen.shuffle(train_index); rand_gen.shuffle(test_index)
        if self.remove_duplicates: 
            train_index = np.unique(train_index)
        return train_index, test_index