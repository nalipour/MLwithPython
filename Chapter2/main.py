import sys
#sys.path.append('../')

import FetchData
import basic_fun as bfun
import matplotlib.pyplot as plt
%matplotlib inline

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelBinarizer
# Waiting for CategoricalEncoder to be implemented in sklearn
from CategoricalEncoder import CategoricalEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import LinearRegression

import pandas as pd

FetchData.fetch_housing_data()
housing = FetchData.load_housing_data()
housing.shape

housing.head()
housing.info()
housing['ocean_proximity'].value_counts()
housing.describe()
housing.hist(bins=50, figsize=(20, 15))
# plt.show()

# ########### Train & Test ########### #
train_set, test_set = bfun.split_train_test(housing, 0.2)
housing_with_id = housing.reset_index()  # adds an 'index' column
train_set, test_set = bfun.split_train_test_by_id(
    housing_with_id, 0.2, 'index')
''' A more unique identifier
    (since longitude and latitude change in millions of years).'''
housing_with_id['id'] = housing['longitude'] * 1000 + housing['latitude']
train_set, test_set = bfun.split_train_test_by_id(housing_with_id, 0.2, 'id')

# Scikit-Learn provides functions to split train/test
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
strat_train_set, strat_test_set = bfun.strat_split(housing, 'median_income', 0.2)


# ######### Visualisation ##########
housing = strat_train_set.copy()
# looks like California
housing.plot(kind='scatter', x='longitude', y='latitude')
# See better places where there is a higher density of data points
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.1)

housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population']/100, label='population', figsize=(10, 7),
             c='median_house_value', cmap=plt.get_cmap('jet'), colorbar=True)
plt.legend()


# ######### Correlations #########
corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
attributes = ['median_house_value', 'median_income', 'total_rooms',
              'housing_median_age']
pd.plotting.scatter_matrix(housing[attributes], figsize=(12, 8))
housing.plot(kind='scatter', x='median_income', y='median_house_value',
             alpha=0.1)

# ---- Attribute combination ---- #
housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population'] / \
    housing['households']

corr_matrix = housing.corr()
corr_matrix['median_house_value'].sort_values(ascending=False)
housing = strat_train_set.drop('median_house_value', axis=1)
# Drop creates a copy of the data and does not affect the train set.
housing_labels = strat_train_set['median_house_value'].copy()

# ######## Data Cleaning ######## #
# total_bedrooms: has some missing values
housing = bfun.data_clean(housing, 'total_bedrooms')

# Take care of missing values using Imputer
housing_tr = bfun.data_clean_imputer(housing, 'median')


# ######### Handling Text and Categorical Attributes #########
housing_cat_encoded = bfun.label_encoder(housing, 'ocean_proximity')

# one-hot encoding
housing_cat_1hot = bfun.label_encode_one_hot(housing_cat_encoded)
# Do in one shot
housing_cat_1hot = bfun.encoding_labelbinarizer(housing['ocean_proximity'])


# ######### Custom Transformers #########

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):  # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self  # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

# ######### Transformation Pipelines ######### #
num_pipeline = Pipeline([
                        ('imputer', Imputer(strategy='median')),
                        ('attribs_adder', CombinedAttributesAdder()),
                        ('std_scaler', StandardScaler())
                        ])

housing_num = housing.drop('ocean_proximity', axis=1)
housing_num_tr = num_pipeline.fit_transform(housing_num)


# To feed directly a DataFrame to the pipeline
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


# One hot encoder
housing_cat = housing['ocean_proximity']
housing_cat_encoded, housing_categories = housing_cat.factorize()

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('label_binarizer', CategoricalEncoder(encoding="onehot-dense"))
])

# Join the two pipelines
full_pipeline = FeatureUnion(transformer_list=[
    ('num_pipeline', num_pipeline),
    ('cat_pipeline', cat_pipeline)
])

# Had to write a CustomLabelBinarizer because the newest version of Scikit crashes
housing_1 = num_pipeline.fit_transform(housing)
housing_1.shape
housing_2 = num_pipeline.fit_transform(housing)
housing_2.shape

housing_prepared = full_pipeline.fit_transform(housing)
housing_prepared.shape

# ######### Training and Evaluating on the Training Set #########
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
np.shape(housing_prepared)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
np.shape(some_data_prepared)
print('Predictions: ', lin_reg.predict(some_data_prepared))
