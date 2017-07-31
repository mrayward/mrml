from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from warnings import warn
from enum import Enum
from matplotlib import pyplot as plt
import pickle
import os
plt.style.use('ggplot')

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


class Cdf:
    '''
    This class serves to calculate the cumulative distribution function and plot it
    '''
    def __init__(self, vals):
        self.vals = vals.copy()
        self.vals.sort()
        self.p = np.linspace(start=0, stop=1, num=len(self.vals))

    def plot(self, ax=None):
        """
        Plots the cdf
        :param ax: matplotlib axis
        :return: plot
        """

        if ax is None:
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(self.p, self.vals)
        ax.set_xlabel('probability $p(x)$')
        ax.set_ylabel('$x$')
        ax.set_title('Cumulative Distribution Function')

    def valuesample(self, n_samples=1, px=None):
        '''
        create a value sample with its probability
        :param n_samples: number of samples
        :param px: probability of the random numbers generated from the sample of size n_samples
        :return:
        '''
        if px is None:
            px = np.random.rand(n_samples)
        return px, np.interp(px, self.p, self.vals)


class MissingValueOptions:
    Keep = 0,
    Replace = 1,
    Delete = 2



def save_pickle(fname, obj):
    '''
    Save pickled file
    :param fname: file name to be saved as
    :param obj: object
    :return: Write a pickled representation of obj to the open file object file
    '''
    with open(fname, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(fname):
    '''
    Loads saved pickled file to work with
    :param fname: file name
    :return: file that was saved as a pickle is now loaded
    '''
    with open(fname, 'rb') as handle:
        b = pickle.load(handle)
    return  b


def split_train(X, y):
    """
    Test Train Split of the data
    :param X: A priori data
    :param y: A posteriori data
    :return: X_train, X_test, y_train, y_test
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

    return X_train, X_test, y_train, y_test


def sequential_train_split(X, y, train_prop=0.75):

    a = int(len(X) * train_prop)
    time_train, time_test = X[0:a], X[a:]
    val_train, val_test = y[0:a], y[a:]

    return time_train, time_test, val_train, val_test


def matricize(s, lookback=3, lookahead=1):
    m = len(s)-lookback-lookahead+1
    xp = np.empty((m, lookback))
    yp = np.empty((m, lookahead))
    idx = np.empty(m, dtype=int)
    for i in range(m):
        xp[i, :] = s[i:i+lookback]
        yp[i, :] = s[i+lookback:i+lookback+lookahead]
        idx[i] = i
    return xp, yp, idx


def df_analysis(df: pd.DataFrame):
    """
    Takes a DataFrame and performs an analysis that gives us information about the type of column and
    problems we may encounter with the different column types such as mixed types and missing values
    :param df: the original data in DataFrame
    :return: a DataFrame with the established parameters for analysis which are found in the 'cols' variable
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
            nan_array = df[cname].isnull()
            row[6] = nan_array.sum() / len(nan_array)
            warn(cname + ' is an object! This is indicative of mixed data types!!')

            # Knowing that the declared type is 'object', we would like to know if it is likely to be numeric
            is_num, row[7], bad_idx = is_the_column_numeric(df, cname, threshold=0.6)

            # We store the non-numeric indices in a dictionary
            dictry[cname] = bad_idx

        # Append every row to a list
        dta.append(row)


    # we ask for it to return a DataFrame, which transforms the list of values we created and identifies the columns,
    # and we also return the dictionary that contains the indices of the values that are non/numeric
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
    """
    This function manages missing values by either Keeping, Deleting or Replacing them
    :param df: Takes in the original DataFrame
    :param c_name: Column in the DataFrame
    :param option: This indicates how we will manage missing values based on the class created above
    :return: A DataFrame with corrected missing values
    """

    values = df[c_name].values

    if option is MissingValueOptions.Keep:
        pass
    elif option is MissingValueOptions.Delete: #find and delete missing values
        # df[c_name].dropna(subset=[c_name], inplace=True)
        # idx = np.where(values != np.nan)[0]
        # df = pd.DataFrame(data=df.values[idx, :], index=df.index.values[idx], columns=df.columns)
        idx = np.where(values == np.nan)[0]
        df.drop(df.index[idx], inplace=True)
    elif option is MissingValueOptions.Replace: #find and replace missing values by filling values forward
        # from : http://pandas.pydata.org/pandas-docs/stable/missing_data.html
        df.fillna(method='pad', inplace=True)


def make_em_nans(df: pd.DataFrame, lst_idx, col_name):
    """
    Makes null values out of mixed types
    :param df: DataFrame
    :param lst_idx: The list of indices where the data is non-numeric in a numeric type column
    :param col_name: Column Name
    :return: The DataFrame with nans where there were non-numeric values in a numeric column
    """

    df[col_name].values[lst_idx] = np.nan

def is_object_numeric(obj):

    numerics = (np.double, np.complex)
    for t in numerics:
        try:
            t(obj)
            return True
        except:
            pass
    return False


def histogram2(df, cname):
    """
    Analysis of a column using a histogram to have a better picture of the values
    :param df: the original DataFrame
    :param cname: Column name
    :return: a DataFrame with the values of the column and their respective frequencies
    """

    # First option is to create a dictionary and store the values as keys and their frequencies as values
    d = dict()
    d['number'] = 0
    for v in df[cname].values:
        if is_object_numeric(v):
            d['number'] += 1

        else:
            if v in d.keys():
                f = d[v]
                f += 1
                d[v] = f
            else:
                d[v] = 1
    df = pd.DataFrame(data=d, columns=d.keys(), index=['freq'])
    return df

def histogram(df, cname):
    lst = df[cname].values
    # Unique Values
    unique_values = df[cname].unique()
    # Sort unique values
    unique_values = unique_values.astype(str)
    unique_values.sort()
    #unique_values = unique_values[::-1]
    n = len(unique_values)
    i = n-1
    number_found = False
    class_mark = list()
    frequency = list()
    ni = 0
    while i > -1 and not is_object_numeric(unique_values[i]):
        idx = np.where(lst==unique_values[i])[0]
        frq = len(idx)
        class_mark.append(unique_values[i])
        frequency.append(frq)
        ni+=frq
        i -= 1
    class_mark.append('numbers')
    frequency.append(len(lst)-ni)
    df = pd.DataFrame(data=frequency, columns=['freq'], index=class_mark)
    return df

def clean(df, missing_value_option: MissingValueOptions=MissingValueOptions.Keep, num_threshold=0.7, str_threshold=0.2,
          string_only_hist = False):
    """
    This function cleans a DataFrame so we can begin working with it as a clean set
    :param df: original DataFrame
    :param missing_value_option: identifies how to manage missing values
    :return: information about the DataFrame
    """

    # Assigning variables to the result of our analysis function
    analysis_result_df, bad_idx_per_column = df_analysis(df)

    # We assign variables to the NaN percentage and the likelihood that the column we are analyzing is numeric
    nan_perc = analysis_result_df['NaN %'].values
    num_likelihood = analysis_result_df['Numerical likelihood'].values

    for c in range(len(analysis_result_df)):

        cname = df.columns.values[c]

        # if analysis_result_df['Type'].values[r] == object:
        if nan_perc[c] == 0 and num_likelihood[c] == 1:
            pass  # It means there are no missing values and it is correct that it is a number column

        elif nan_perc[c] == 0 and (0 < num_likelihood[c] < 1): # It means there are no missing values, but the
            # column has mixed data types

            bad_idx = bad_idx_per_column[cname]

            if num_likelihood[c] > num_threshold:
                print('\t Inconsistent number column!', cname)
                # even though up to 30% of the data in this column is garbage, the majority of the data in this
                # column is numeric, so we will throw away the rest of the values
                if missing_value_option is not MissingValueOptions.Keep:
                    make_em_nans(df=df, lst_idx=bad_idx, col_name=cname)
                    correct_missing_values(df, c_name=cname, option=missing_value_option)
                else:
                    pass
            else:
                print('\t Inconsistent string column!', cname, ' %of numbers: ', num_likelihood[c])
                hist_df = histogram(df, cname)
                print(hist_df)
                hist_df.plot(kind='bar', title=cname)

        elif nan_perc[c] > 0 and num_likelihood[c] == 1:  # There are missing values, but the data type is consistent
            print('There are missing values in this column:', cname)
            correct_missing_values(df, c_name=cname, option=missing_value_option)

        elif nan_perc[c] > 0 and (0 < num_likelihood[c] < 1):
            print('There are missing values AND inconsistent data!', cname)
            if num_likelihood[c] < str_threshold:
                hist_df = histogram(df, cname)
                print(hist_df)
                hist_df.plot(kind='bar', title=cname)
            else:
                print('\t This column is messed up because it has numbers and strings. Advice: Check it out!')

        elif nan_perc[c] == 0 and num_likelihood[c] == 0:
            print('This is likely a string column', cname)
            if string_only_hist:
                hist_df = histogram(df, cname)
                print(hist_df)
                hist_df.plot(kind='bar', title=cname)

            else:
                pass

        else:
            print('There is something really wrong!', cname)

    return analysis_result_df


class MrProper:
    '''
    Cleaning Class
    '''

    def __init__(self, df: pd.DataFrame, missing_value_option: MissingValueOptions=MissingValueOptions.Keep,
                 numericthreshold = 0.7, stringthreshold = 0.2, stringonlyhist=False):
        '''
        Initialize the class
        :param df: DataFrame
        :param missing_value_option: How to manage missing values
        :param numericthreshold: Threshold where we consider the column to be numeric
        :param stringthreshold: Percentage it cannot surpass in order for us consider the column to be a string
        :param stringonlyhist: Histogram for string values
        '''
        self.cleandf = df.copy()
        self.missing_value_option = missing_value_option
        self.numericthreshold = numericthreshold
        self.stringthreshold = stringthreshold
        self.stringonlyhist = stringonlyhist
        self.prioranalysis = None
        self.numerical_trustworthy_columns = None

    def clean(self):
        '''
        Clean method
        :return: modifies df to the clean version
        '''
        # self.cleandf will be modified by this function
        self.prioranalysis = clean(self.cleandf,
                                   self.missing_value_option,
                                   self.numericthreshold,
                                   self.stringthreshold,
                                   self.stringonlyhist)
        self.numerical_trustworthy_columns= self.prioranalysis[self.prioranalysis['Numerical likelihood'] == 1.0].index.values

    def numerical_trustworthy_df(self):
        '''
        cleaning numerical columns
        :return: clean numerical trustworthy columns
        '''
        return self.cleandf[self.numerical_trustworthy_columns]



if __name__ == '__main__':

    f = os.path.join('cancer', 'Wisconsin_breast_cancer_wrong.csv')

    data_ = pd.read_csv(f)

    clean(data_, MissingValueOptions.Replace)

    print(data_.head(20))

    df_ref, _ = df_analysis(data_)
    print(df_ref)

    plt.show()