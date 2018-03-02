import sys
sys.path.append('../')

import FetchData
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np

FetchData.fetch_housing_data()
housing = FetchData.load_housing_data()

housing.head()
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()

housing.hist(bins=50, figsize=(20, 15))
# plt.show()


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


train_set, test_set = split_train_test(housing, 0.2)
