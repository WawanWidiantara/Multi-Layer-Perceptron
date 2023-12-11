# %%
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

# %% [markdown]
# # Loading and Getting a Glimpse of Dataset

# %%
data = pd.read_excel(
    r"D:\Code\py_code\Multi-Layer-Perceptron\data\BA_AirlineReviews_CL_excel.xlsx",
    header=0,
)

data.head()

# %%
data.info()

# %% [markdown]
# # Preprocessing Data

# %% [markdown]
# **Checking missing value**

# %%
data.isnull().sum()

# %% [markdown]
# **Replace Missing Value**

# %% [markdown]
# 1. Missing Value Numerical Data

# %%
numeric_missing_col = data.columns[
    (data.isnull().any()) & (data.dtypes != "object") & (data.columns != "DateFlown")
].to_list()
numeric_missing_col

# %%
numeric = data[numeric_missing_col].values

impNumeric = SimpleImputer(strategy="constant", fill_value=0)
impNumeric = impNumeric.fit(numeric)
numeric = impNumeric.transform(numeric)
data[numeric_missing_col] = numeric

data.isnull().sum()

# %% [markdown]
# 2. Missing Value Nominal Data

# %%
nominal_missing_col = data.columns[
    (data.isnull().any()) & (data.dtypes == "object")
].to_list()
nominal_missing_col.append("DateFlown")
nominal_missing_col

# %%
nominal = data[nominal_missing_col].values

impNominal = SimpleImputer(strategy="constant", fill_value="unknown")
impNominal = impNominal.fit(nominal)
nominal = impNominal.transform(nominal)
data[nominal_missing_col] = nominal

data.isnull().sum()

# %% [markdown]
# **Removing Unwanted Feature**

# %%
data.drop(
    ["id", "Name", "Datetime", "DateFlown", "ReviewHeader", "ReviewBody"],
    inplace=True,
    axis=1,
)
data.head()


# %%
encoder = OneHotEncoder(sparse=False)

encoded_data = encoder.fit_transform(data[["Satisfaction"]])
encoded_df = pd.DataFrame(
    encoded_data, columns=encoder.get_feature_names_out(["Satisfaction"])
)
encoded_df.head()

# # %% [markdown]
# # **Feature Encoding**

# # %%
# lbenc = LabelEncoder()

# for i in data.columns.values:
#     if (data[i].dtypes == "object" or data[i].dtypes == "bool") and i != "Satisfaction":
#         data[i] = lbenc.fit_transform(data[i].astype(str))

# # %%
# satisfaction_order = [
#     "Very Dissatisfied",
#     "Dissatisfied",
#     "Neutral",
#     "Satisfied",
#     "Very Satisfied",
#     "Enthusiastic",
#     "Extremely Satisfied",
#     "Delighted",
#     "Evangelist",
#     "Advocate",
# ]

# ordinal_enc = OrdinalEncoder(categories=[satisfaction_order])
# data["Satisfaction"] = ordinal_enc.fit_transform(data[["Satisfaction"]])

# # %%
# data.head()

# # %% [markdown]
# # **Splitting the Data**

# # %%
# import pandas as pd
# from sklearn.model_selection import train_test_split

# # Assuming data is a DataFrame with features in columns 1 and onwards, and labels in the first column
# features = data.iloc[:, 1:].values
# label = data.iloc[:, 0].values

# # Split the data into training and testing sets with stratification
# x_train, x_test, y_train, y_test = train_test_split(
#     features, label, test_size=0.1, random_state=42, stratify=data["Satisfaction"]
# )

# class_1_ratio = 0.7
# class_2_ratio = 0.3

# num_class_1_instances = int(len(y_train) * class_1_ratio)
# num_class_2_instances = int(len(y_train) * class_2_ratio)
# num_class_1_instances_test = int(len(y_test) * class_1_ratio)
# num_class_2_instances_test = int(len(y_test) * class_2_ratio)

# # Convert y_train to a pandas Series to use iloc
# y_train_series = pd.Series(y_train)
# y_test_series = pd.Series(y_test)

# # Extract indices of class 1 and class 2 instances
# class_1_indices = y_train_series[y_train_series.isin([0, 1])].index
# class_2_indices = y_train_series[~y_train_series.isin([0, 1])].index
# class_1_indices_test = y_test_series[y_test_series.isin([0, 1])].index
# class_2_indices_test = y_test_series[~y_test_series.isin([0, 1])].index

# # Sample instances based on the calculated number of instances for each class
# selected_class_1_indices = class_1_indices[:num_class_1_instances]
# selected_class_2_indices = class_2_indices[:num_class_2_instances]
# selected_class_1_indices_test = class_1_indices_test[:num_class_1_instances_test]
# selected_class_2_indices_test = class_2_indices_test[:num_class_2_instances_test]

# # Combine indices of selected instances for both classes
# selected_indices = selected_class_1_indices.union(selected_class_2_indices)
# selected_indices_test = selected_class_1_indices_test.union(
#     selected_class_2_indices_test
# )

# # Use the selected indices to create the final training set
# X_train_final = x_train[selected_indices]
# y_train_final = y_train[selected_indices]
# X_test_final = x_test[selected_indices_test]
# y_test_final = y_test[selected_indices_test]

# X_train_final.shape, y_train_final.shape, X_test_final.shape, y_test_final.shape

# # %% [markdown]
# # **Feature Scaling**
