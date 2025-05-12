import pandas as pd
import numpy as np

df = pd.read_csv('l1_day.csv')

df['date'] = pd.to_datetime(df['date'])
df['date'] = df['date'].dt.strftime('%Y-%m-%d')

df = df.set_index('date')

print(df.head())
