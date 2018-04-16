import numpy as np
import hashlib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer


# ########### Train & Test ########### #


def split_train_test(data, test_ratio):
    '''  Split train and test set randomly '''
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio, hash):
    ''' Use a hash to select train and test sets '''
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio


def split_train_test_by_id(data, test_ratio, id_column, hash=hashlib.md5):
    ''' If new data gets appended, it might happen that the ID
        is not unique anymore (so we should be careful) '''
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
    return data.loc[~in_test_set], data.loc[in_test_set]


def strat_split(housing, variable, test_ratio):
    ''' Median income: important attribute to predict housing price
    Make categories and all of them must represent equally (strats)
    Merge together cateegories greater than 5.0
    (implementation is totally counterintuitive?) '''
    housing['income_cat'] = np.ceil(housing[variable]/1.5)
    housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
    housing['income_cat'].hist()
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    housing['income_cat'].value_counts()/len(housing)
    # Remove the income categories
    for set_ in (strat_train_set, strat_test_set):
        set_.drop('income_cat', axis=1, inplace=True)

    return strat_train_set, strat_test_set


# ########### Data Cleaning ########### #

def data_clean(df, param):
    ''' Missing parameters: param are replaced by the median'''
    df.dropna(subset=[param])
    df.drop(param, axis=1)
    median = df[param].median()
    df[param].fillna(median, inplace=True)
    return df


def data_clean_imputer(df, my_strategy):
    imputer = Imputer(strategy=my_strategy)
    df_num = df.drop('ocean_proximity', axis=1)
    imputer.fit(df_num)
    imputer.statistics_
    imputer.strategy

    df_num.median().values
    X = imputer.transform(df_num)
    df_tr = pd.DataFrame(X, columns=df_num.columns)
    df_num.median().values
    return df_tr


# ######### Handling Text and Categorical Attributes ######### #

def label_encoder(df, var):
    encoder = LabelEncoder()
    df_cat = df[var]
    df_cat_encoded = encoder.fit_transform(df_cat)
    print(encoder.classes_)
    return df_cat_encoded


def label_encode_one_hot(encoded):
    ''' One hot encoding '''
    encoder = OneHotEncoder()
    df_cat_1hot = encoder.fit_transform(encoded.reshape(-1, 1))
    df_cat_1hot.toarray()

    return df_cat_1hot


def encoding_labelbinarizer(df_cat):
    ''' Encoding in one shot '''
    encoder = LabelBinarizer()
    df_cat_1hot = encoder.fit_transform(df_cat)
    return df_cat_1hot

# ######### Custom Transformers #########
