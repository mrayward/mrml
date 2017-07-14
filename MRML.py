from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from warnings import warn
from enum import Enum
import os

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class MissingValueOptions:
    Keep = 0,
    Replace = 1,
    Delete = 2


def split_train(X, y):
    """
    Test Train Split of the data
    :param X: A priori data
    :param y: A posteriori data
    :return: X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    return X_train, X_test, y_train, y_test


def df_analysis(df: pd.DataFrame):
    """
    Takes a dataframe and performs
    :param df:
    :return:
    """
    # Creating a list to append the rows of data, which will be later transformed into a DataFrame
    dta = list()

    # These are the new columns of the resulting DataFrame
    cls = ['Type', 'General', 'Max', 'Min', 'Mean', 'Range', 'NaN %', 'Numerical likelihood']

    # The number of columns that our analysis will contain (detailed above)
    cols = len(cls)

    # This is the dictionary we create to store the column name and the values that are non-numeric within it
    dictry = dict()

    # These are all the categories of numbers we are considering, in order to classify a type as numeric
    numerics = (np.int16, np.int32, np.int64,
                np.float16, np.float32, np.float64,
                np.double, np.complex)

    # for every column in the df:
    for cname in df.columns.values:

        # Declared empty row with 'cols' number of values
        row = [None] * cols
        # For the analysis function, we need to know the type of column
        tpe = df[cname].dtype

        row[0] = tpe

        if tpe in numerics:  # checking to see if type is numerical
            nan_array = df[cname].isnull()
            row[1] = "I'm a number"
            row[6] = nan_array.sum() / len(nan_array)
            # non_nan_array = 1-nan_array
            idx = np.where(nan_array==False)[0]
            row[2] = df[cname].values[idx].max()
            row[3] = df[cname].values[idx].min()
            row[4] = row[2] - row[3]
            row[5] = df[cname].values[idx].mean()
            row[7] = 1

        elif isinstance(tpe, str):  # checking to see if type is a string
            row[1] = "I'm a string"

            df2 = df[(df[cname] == ' ') | (df[cname] == '')]
            if len(df2) > 0:
                warn(cname + ' (string) contains missing values!')
                row[6] = len(df2) / len(df[cname])
            row[7] = 0

        elif isinstance(tpe, bool):  # checking to see if it is a boolean
            '''
            If pandas marked this column as boolean, there are no missing values,
            because everything is either a 0 or a 1.
            '''
            row[1] = "I'm a boolean"
            row[7] = 0

        else:  # if it is not any of the above types, we conclude it is an object
            row[1] = "I'm a object"
            row[2] = 0
            row[3] = 0
            row[4] = 0
            row[5] = 0
            row[6] = 0
            warn(cname + ' is an object! This is indicative of mixed data types!!')

            # Knowing that the declared type is 'object', we would like to know if it is likely to be numeric
            is_num, row[7], bad_idx = is_the_column_numeric(df, cname, threshold=0.6)

            # We store the non-numeric indices in a dictionary
            dictry[cname] = bad_idx

        # Append every row to a list
        dta.append(row)


    return pd.DataFrame(data=dta, columns=cls, index=df.columns.values), dictry


def is_the_column_numeric(df, col_name, threshold=0.7):
    """
    Determines whether a column is numeric
    :param df: DataFrame
    :param col_name: name of the column we would like to analyze within the DataFrame
    :param threshold: minimum proportion of numbers that the column should have in order to be considered numeric
    :return: True or False, the proportion and the indices where the column is not numeric
    """
    values = df[col_name].values
    n = len(values)
    num_counter = 0
    indx = list()

    for i, v in enumerate(values):

        try:
            float(v)
            num_counter += 1
        except:
            try:
                complex(v)
                num_counter += 1
            except:
                # if I reach this point, it means this is not a number
                indx.append(i)

    p = num_counter / n

    if p > threshold:
        return True, p, indx
    else:
        return False, p, indx


def correct_missing_values(df: pd.DataFrame, c_name, option: MissingValueOptions):

    values = df[c_name].values

    if option is MissingValueOptions.Keep:
        pass
    elif option is MissingValueOptions.Delete:
        # df[c_name].dropna(subset=[c_name], inplace=True)
        # idx = np.where(values != np.nan)[0]
        # df = pd.DataFrame(data=df.values[idx, :], index=df.index.values[idx], columns=df.columns)
        idx = np.where(values == np.nan)[0]
        df.drop(df.index[idx], inplace=True)
    elif option is MissingValueOptions.Replace:
        # from : http://pandas.pydata.org/pandas-docs/stable/missing_data.html
        df.fillna(method='pad', inplace=True)


def make_em_nans(df: pd.DataFrame, lst_idx, col_name):

    # v = df[col_name].values
    # v[lst_idx] = np.nan
    df[col_name].values[lst_idx] = np.nan


def clean(df, missing_value_option: MissingValueOptions=MissingValueOptions.Keep):

    analysis_result_df, bad_idx_per_column = df_analysis(df)

    print(analysis_result_df)

    nan_perc = analysis_result_df['NaN %'].values
    num_likelihood = analysis_result_df['Numerical likelihood'].values

    for c in range(len(analysis_result_df)):

        cname = df.columns.values[c]

        # if analysis_result_df['Type'].values[r] == object:
        if nan_perc[c] == 0 and num_likelihood[c] == 1:
            pass  # It means there are no missing values and it is correct that it is a number column

        elif nan_perc[c] == 0 and (0 < num_likelihood[c] < 1):
            print('Inconsistent data! Do SOMETHING!', cname)
            # TODO: implement method
            bad_idx = bad_idx_per_column[cname]

            if num_likelihood[c] > 0.7:
                print('\t Inconsistent number column!')
                # even though up to 30% of the data in this column is garbage, the majority of the data in this
                # column is numeric, so we will throw away the rest of the values
                if missing_value_option is not MissingValueOptions.Keep:
                    make_em_nans(df=df, lst_idx=bad_idx, col_name=cname)
                    correct_missing_values(df, c_name=cname, option=missing_value_option)
                else:
                    pass
            else:
                print('\t Inconsistent string column!')  # TODO: implement method

        elif nan_perc[c] > 0 and num_likelihood[c] == 1:
            print('There are missing values in this column:', cname)
            correct_missing_values(df, c_name=cname, option=missing_value_option)

        elif nan_perc[c] > 0 and (0 < num_likelihood[c] < 1):
            print('There are missing values AND inconsistent data!', cname)
            # TODO: implement methods

        elif nan_perc[c] == 0 and num_likelihood[c] == 0:
            print('This is likely a string column', cname)

        else:
            print('There is something really wrong!', cname)

    print()


if __name__ == '__main__':

    f = os.path.join('cancer', 'Wisconsin_breast_cancer_wrong.csv')

    data_ = pd.read_csv(f)

    clean(data_, MissingValueOptions.Replace)

    print(data_.head(20))

    df_ref, _ = df_analysis(data_)
    print(df_ref)