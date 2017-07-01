from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from warnings import warn
import os

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def split_train(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    return X_train, X_test, y_train, y_test


def df_analysis(df: pd.DataFrame):

    dta = list()

    numerics = (np.int16, np.int32, np.int64,
                np.float16, np.float32, np.float64,
                np.double, np.complex)

    for cname in df.columns.values:

        row = [None]*7

        tpe = df[cname].dtype

        row[0] = tpe

        if tpe in numerics:
            nan_array = df[cname].isnull()
            row[1] = "I'm a number"
            row[6] = nan_array.sum() / len(nan_array)
            # non_nan_array = 1-nan_array
            idx = np.where(nan_array == False)[0]
            row[2] = df[cname].values[idx].max()
            row[3] = df[cname].values[idx].min()
            row[4] = row[2] - row[3]
            row[5] = df[cname].values[idx].mean()

        elif isinstance(tpe, str):
            row[1] = "I'm a string"

            df2 = df[df[cname] == ' ' | df[cname] == '']
            if len(df2) > 0:
                warn(cname + ' (string) contains missing values!')
                row[6] = len(df2) / len(df[cname])

        elif isinstance(tpe, bool):
            '''
            If pandas marked this column as boolean, there are no missing values,
            because everything is either a 0 or a 1.
            '''
            row[1] = "I'm a boolean"
        else:
            row[1] = "I'm a object"
            warn(cname + ' is an object! This is indicative of mixed data types!!')

        dta.append(row)

    cls = ['Type', 'General', 'Max', 'Min', 'Mean', 'Range', 'NaN %']
    return pd.DataFrame(data=dta, columns=cls, index=df.columns.values)


def correct_missing_values(df):

    values = df.values

    # look for NaN
    idx = np.where(values == np.nan)

    if len(idx[0]) > 0 or len(idx[1]) > 0:
        # nan found
        warn('nan found')



def clean(df):
    print()


if __name__ == '__main__':

    f = os.path.join('cancer', 'Wisconsin_breast_cancer.csv')

    data_ = pd.read_csv(f)

    df_res = df_analysis(data_)
    print(df_res)