from market.yahoo_finance import load_yahoo_quote
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import datetime
from MRML import sequential_train_split, matricize
from sklearn.linear_model import LinearRegression

start = datetime.datetime(2000, 1, 1)
end = datetime.datetime.today()

apple_df = load_yahoo_quote('FB', start, end)
print(apple_df)
X = apple_df.index
y = apple_df['Close'].values

lk_bck = 20
xtrain, ytrain, itrain = matricize(y, lookback=lk_bck)

model_ = LinearRegression()
model_.fit(xtrain, ytrain)
ypredict = model_.predict(y[-lk_bck:])

print('Magic=', ypredict)

from twitter.api import Twitter, NoAuth

auth = NoAuth()
twitter = Twitter('miriamrayward', '19115st')
twitter.statuses.update(status="Tomorrow's Facebook closing price is $" + str(ypredict))

