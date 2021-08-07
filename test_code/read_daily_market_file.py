import pandas as pd


df = pd.read_csv(r'C:\PycharmProjects\PorfolioMonitor\porfolio_files\ZZ500VG.csv')
df.to_excel(r'C:\PycharmProjects\PorfolioMonitor\porfolio_files\ZZ500VG.xlsx',index=False)