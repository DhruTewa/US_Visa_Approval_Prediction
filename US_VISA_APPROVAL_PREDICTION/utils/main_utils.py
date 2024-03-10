import os
import sys
import pandas as pd
from pandas import DataFrame
import numpy as np
import dill
import yaml
from US_VISA_APPROVAL_PREDICTION.exception import USvisaException
from US_VISA_APPROVAL_PREDICTION.logger import logging


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)

    except Exception as e:
        raise USvisaException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise USvisaException(e, sys) from e


def load_object(file_path: str) -> object:
    logging.info("Entered the load_object method of utils")

    try:

        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)

        logging.info("Exited the load_object method of utils")

        return obj

    except Exception as e:
        raise USvisaException(e, sys) from e

def save_numpy_array_data(file_path: str, array: np.array):
    """
    Save numpy array data to file
    file_path: str location of file to save
    array: np.array data to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise USvisaException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """
    load numpy array data from file
    file_path: str location of file to load
    return: np.array data loaded
    """
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise USvisaException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    logging.info("Entered the save_object method of utils")

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

        logging.info("Exited the save_object method of utils")

    except Exception as e:
        raise USvisaException(e, sys) from e


def drop_columns(df: DataFrame, cols: list)-> DataFrame:

    """
    drop the columns form a pandas DataFrame
    df: pandas DataFrame
    cols: list of columns to be dropped
    """
    logging.info("Entered drop_columns methon of utils")

    try:
        df = df.drop(columns=cols, axis=1)

        logging.info("Exited the drop_columns method of utils")
        
        return df
    except Exception as e:
        raise USvisaException(e, sys) from e
    
    
# function to read the data file


def read_file_to_dataframe(path):
    dataset = pd.read_csv(path)
    return dataset

# function to perform initial analysis on the data 

def initial_analysis(dataset):
    #shape of the dataset
    
    print("\n The shape of the dataset:{}".format(dataset.shape))
    print("\n Number of columns in the dataset:{}".format(dataset.shape[1]))
    print("\n Number of rows in the dataset:{}".format(dataset.shape[0]))
    print("\n Data types in the dataset :\n \n{}".format(dataset.dtypes))
    print("*"*45)
    
       
    # Numerical and Categorical feautres in the dataset
    
    # define numerical & categorical columns
    numeric_features = [feature for feature in dataset.columns if dataset[feature].dtype != 'O']
    categorical_features = [feature for feature in dataset.columns if dataset[feature].dtype == 'O']

# print columns
    print('We have {} numerical features : {}'.format(len(numeric_features), numeric_features))
    print('\nWe have {} categorical features : {}'.format(len(categorical_features), categorical_features))
    print("*"*40)
    
    # Missing values
    print(" Number of missing values :\n \n{}".format(dataset.isna().sum()))
    print("*"*40)
    
    # Checking the duplicates
    print("\n**The dataset has {} duplicate rows.**".format(dataset.duplicated().sum()))
    
    return initial_analysis

# a python function to create list of numerical and categorical variaable

def get_variables(df):
    #date_vars = []
    num_vars = []
    cat_vars = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            num_vars.append(col)
        elif pd.api.types.is_string_dtype(df[col]):
            cat_vars.append(col)

    return num_vars, cat_vars

#def outlier_analysis:
import matplotlib.pyplot as plt
plt.figure(figsize=(16,6))
plt.show()

# function to find the outliers

def outlier_detection(dataset,i):
    stats=dataset[i].describe()
    IQR = stats['75%']-stats['25%'] # Finding the inter quartile range 
    upper=stats['75%']+1.5*IQR      # upper cut-off limit
    lower=stats['25%']-1.5*IQR      # lower cut-off limit
    print('For feature {}. Upper bound : {:.2f} Lower Bound: {:.2f}'.format(i,upper,lower))

def handle_outliers_missing_values(df):
    # Handle missing values
    df.fillna(df.mean(), inplace=True)

    # Handle outliers
    for col in df.select_dtypes(include=['int', 'float']).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    return df

def create_subsets(df, column_name):
    """
    Returns a list of individual DataFrames, one for each unique value in the specified column.

    Parameters:
    df (pandas.DataFrame): The DataFrame to subset.
    column_name (str): The name of the column to filter on.

    Returns:
    list: A list of individual DataFrames.
    """
    subset_list = []
    for column_value in df[column_name].unique():
        subset = df[df[column_name] == column_value].copy()
        subset_name = f"{column_name}_{column_value}"
        subset.name = subset_name
        subset_list.append(subset)
    return subset_list

def correlation(dataset,threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > threshold:
                colname = corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr