import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

d = {'no':False, 'No':False, 'NO':False, 'SI':True, 'si':True, 'Si':True}

pacientes = pd.read_excel('/Users/miriamrayward/Desktop/Informacion_Pacientes.xlsx', sheetname='Pacientes_Internacionales')
pacientes['Ciclo?'] = pacientes['Ciclo?'].str.strip().fillna(False).replace(d)
pacientes['Aparecio?'] = pacientes['Aparecio?'].str.strip().fillna(False).replace(d)


viajes = pd.read_excel('/Users/miriamrayward/Desktop/Informacion_Pacientes.xlsx', sheetname='Viajes')
viajes['Fecha_Trip'] = pd.to_datetime(viajes['Fecha_Trip'])
viajes.rename(columns={'Medico':'Medico_viaje' }, inplace=True)

df = pd.merge(pacientes, viajes, how='inner', on='Trip_ID')

# 1. Numero de person
c1 = df.groupby(['Ciudad_Trip', 'Fecha_Trip'])['Trip_ID'].count()


for city in c1.index.get_level_values('Ciudad_Trip').unique().values:
    plt.figure(figsize=(10, 6))
    c1[c1.index.get_level_values('Ciudad_Trip')==city].plot(kind='bar')
    plt.title(city)


# aparecio si\no
c2 = df.groupby(['Ciudad_Trip', 'Fecha_Trip'])['Aparecio?'].count()
c2.rename(columns={'Aparecio?':'count'}, inplace=True)
c21 = df[df['Aparecio?']==True].groupby(['Ciudad_Trip', 'Fecha_Trip'])['Aparecio?'].count()
c21.rename(columns={'Aparecio?':'yes'}, inplace=True)
result = pd.concat([c2, c21], axis=1, join='inner')
result['proportion'] = result[1] / result[0]
result.to_excel('aparecio.xlsx')

for city in c2.index.get_level_values('Ciudad_Trip').unique().values:
    plt.figure(figsize=(10, 6))
    c2[c2.index.get_level_values('Ciudad_Trip')==city].plot(kind='bar', color='r')
    c21[c21.index.get_level_values('Ciudad_Trip') == city].plot(kind='bar')
    plt.title(city)

#3. De las que se apuntaron, cuantas se hicieron ciclo
c3 = df.groupby(['Ciudad_Trip', 'Fecha_Trip'])['Aparecio?'].count()
c3.rename(columns={'Aparecio?':'count'}, inplace=True)
c31 = df[(df['Aparecio?']==True) & (df['Ciclo?']==True)].groupby(['Ciudad_Trip', 'Fecha_Trip'])['Aparecio?'].count()
c31.rename(columns={'Aparecio?':'yes'}, inplace=True)
result3 = pd.concat([c3, c31], axis=1, join='inner')
result3['proportion'] = result3[1] / result3[0]
result3.to_excel('apuntaronyciclo.xlsx')

for city in result3.index.get_level_values('Ciudad_Trip').unique().values:
    plt.figure(figsize=(10, 6))
    result3[result3.index.get_level_values('Ciudad_Trip')==city][1].plot(kind='bar', color='r')
    plt.title('Tratamientos en ' + city)

# 4. de las que aparecieron, cuantas se hicieron ciclo
c4 = df[df['Aparecio?']==True].groupby(['Ciudad_Trip', 'Fecha_Trip'])['Aparecio?'].count()
c4.rename(columns={'Aparecio?':'count'}, inplace=True)
c41 = df[(df['Aparecio?']==True) & (df['Ciclo?']==True)].groupby(['Ciudad_Trip', 'Fecha_Trip'])['Aparecio?'].count()
c41.rename(columns={'Aparecio?':'yes'}, inplace=True)
result4 = pd.concat([c4, c41], axis=1, join='inner')
result4['proportion'] = result4[1] / result4[0]
result4.to_excel('aparecioyciclo.xlsx')

print(df.head(10))
print(c21)
print(result)


plt.show()