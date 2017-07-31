from market.yahoo_finance import load_yahoo_quote
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
from MRML import sequential_train_split
from sklearn.linear_model import LinearRegression

start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.today()

apple_df = load_yahoo_quote('FB', start, end)
print(apple_df)
X = apple_df.index
y = apple_df['Close'].values


time_train, time_test, val_train, val_test = sequential_train_split(X, y)

# Model 1: Baseline
model = LinearRegression()
model.fit(time_train.astype(np.int64).values.reshape(-1, 1), val_train.reshape(-1, 1))

val_predicted = model.predict(time_test.astype(np.int64).values.reshape(-1, 1))

plt.plot(time_train, val_train, c='r')
plt.plot(time_test, val_test, c='b')
plt.plot(time_test, val_predicted, c='k')


# better model hopefully...
def matricize(s, lookback=3, lookahead=1):
    m = len(s)-lookback-lookahead+1
    xp = np.empty((m, lookback))
    yp = np.empty((m, lookahead))
    idx = np.empty(m, dtype=int)
    for i in range(m):
        xp[i,:] = s[i:i+lookback]
        yp[i,:] = s[i+lookback:i+lookback+lookahead]
        idx[i] = i
    return xp, yp, idx

xtrain, ytrain, itrain = matricize(val_train, lookback=20)
xtest, ytest, itest = matricize(val_test, lookback=20)

model = LinearRegression()
model.fit(xtrain, ytrain)

ypredict = model.predict(xtest)
plt.plot(time_test[itest+20], ypredict, c='g')

print("Tomorrow's price is: ", ypredict[-1])

plt.show()


