# http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime

import pandas as pd
import numpy as np
# import MRML as mrml
from MRML import *
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.tree                   import DecisionTreeRegressor
from sklearn.neighbors              import KNeighborsRegressor
from sklearn.svm                    import SVR
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor

pickle_name = 'mrproper.pkl'

if os.path.exists(pickle_name):

    mr_proper = load_pickle(pickle_name)

else:
    df = pd.read_csv('crime2.csv')
    print(df.head(10))

    mr_proper = MrProper(df)
    mr_proper.clean()
    print(mr_proper.prioranalysis)
    print(mr_proper.numerical_trustworthy_columns)

    save_pickle(pickle_name, mr_proper)



df2 = mr_proper.numerical_trustworthy_df()

y = df2['ViolentCrimesPerPop '].values
X = df2.drop(['ViolentCrimesPerPop '], axis=1).values


pca = PCA(n_components=15)
Xpca = pca.fit_transform(X, y)

print(Xpca.shape)

X_train, X_test, y_train, y_test = split_train(Xpca, y)




models = [LinearRegression(), DecisionTreeRegressor(), KNeighborsRegressor(), SVR(), MLPRegressor((100, 100,))]

for model in models:
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    score = model.score(X_test, y_test)
    print(score)




# print(df2.cov())
# df2.cov().to_excel('crime2_cov.xls')

# df_clean = df.copy()

# analysis_df = clean(df_clean, MissingValueOptions.Replace)
# print(analysis_df)


# df_ref, _ = df_analysis(df)
# print(df_ref)

