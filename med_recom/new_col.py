import pandas as pd

df = pd.read_csv("data/data.csv")
df['col_for_ocr'] = df['Medicine Name'] + " " + df['Composition'] + " " + df['Manufacturer'].astype(str)
print(df['col_for_ocr'].head())

df.to_csv('med_data',index=False)
