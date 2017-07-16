# http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime

import pandas as pd
import numpy as np
# import MRML as mrml
from MRML import *


df = pd.read_csv('crime2.csv')
print(df.head(10))

mr_proper = MrProper(df)
mr_proper.clean()
print(mr_proper.prioranalysis)
print(mr_proper.numerical_trustworthy_columns)

df2 = mr_proper.numerical_trustworthy_df()

print(df2.cov())
df2.cov().to_excel('crime2_cov.xls')

# df_clean = df.copy()

# analysis_df = clean(df_clean, MissingValueOptions.Replace)
# print(analysis_df)


# df_ref, _ = df_analysis(df)
# print(df_ref)

