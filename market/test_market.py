from market.yahoo_finance import load_yahoo_quote
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
from MRML import sequential_train_split, matricize
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import  KNeighborsRegressor
from sklearn.svm import SVR, LinearSVR

start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.today()

apple_df = load_yahoo_quote('AAPL', start, end)
print(apple_df)
X = apple_df.index
y = apple_df['Close'].values


time_train, time_test, val_train, val_test = sequential_train_split(X, y)

# Model 1: Baseline
model = LinearRegression()
model.fit(time_train.astype(np.int64).values.reshape(-1, 1), val_train.reshape(-1, 1))

val_predicted = model.predict(time_test.astype(np.int64).values.reshape(-1, 1))

plt.plot(time_train, val_train, c='r', label='Real train price')
plt.plot(time_test, val_test, c='b', label='Real test price')
# plt.plot(time_test, val_predicted, c='k', label='Linear regression forecast')

def mse(a, b):
    return np.sqrt(np.power(a-b, 2)).mean()

score_model = mse(val_test, val_predicted.reshape(-1))
print('Baseline score = ', score_model)

# better model hopefully...
lk_bck = 20
xtrain, ytrain, itrain = matricize(val_train, lookback=lk_bck)
xtest, ytest, itest = matricize(val_test, lookback=lk_bck)

models = [LinearRegression(),
          KNeighborsRegressor(n_neighbors=1, p=10.5),
          RandomForestRegressor(),
          MLPRegressor(hidden_layer_sizes=(100, 100, 100), activation="relu", solver='adam', alpha=0.0001,
                       batch_size=1, learning_rate="constant", learning_rate_init=0.001, power_t=0.5,
                       max_iter=20, shuffle=True, random_state=None, tol=1e-8, verbose=True, warm_start=False,
                       momentum=0.1, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1,
                       beta_1=0.9, beta_2=0.999, epsilon=1e-8),
          SVR(),
          LinearSVR()
          ]
model_names = ['Linear Regression', 'KNN', 'Random Forest', 'Neural Network', 'SVR', 'LinearSVR']
colors = ['g', 'c', 'm', 'k', 'y', 'orange']
for i, model_ in enumerate(models):

    model_.fit(xtrain, ytrain)
    ypredict = model_.predict(xtest)

    scre = mse(ytest, ypredict)
    lbl = model_names[i] + ' forecast, ' + str(scre) + ' €'
    plt.plot(time_test[itest+lk_bck], ypredict, c=colors[i], label=lbl)

    print(model_names[i], 'score = ', scre)

# More models to compare


plt.legend(loc='best')
plt.show()


