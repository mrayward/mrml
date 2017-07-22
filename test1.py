from MRML import histogram
import pandas as pd


df = pd.DataFrame(data=[1,2,4,1,3,'Juan',3,'Pepe','Juan',0,2], columns=['col1'])

hist = histogram(df, 'col1')

print(hist)