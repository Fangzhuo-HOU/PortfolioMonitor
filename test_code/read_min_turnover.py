import pandas as pd
import os

porfolios = os.listdir(r'C:\PycharmProjects\PorfolioMonitor\porfolio_files')
porfolios_file = pd.DataFrame()
for i in porfolios:
    temp_df = pd.read_csv(r'C:\PycharmProjects\PorfolioMonitor\porfolio_files' + '\\' + i)
    temp_df['flag'] = str(i.split('.')[0])
    temp_df['base'] = str(i[:2])
    porfolios_file = porfolios_file.append(temp_df)

# 读入每只股票的min bar信息
stkcds = pd.DataFrame(porfolios_file['stkcd'])
stkcds['stkcd'] = stkcds.apply(lambda x: str(x['stkcd'])[2:] + '.' + str(x['stkcd'])[:2].upper(), axis=1)
stkcds = stkcds['stkcd'].values

# minBar database path
min_bar_path = r'C:\Users\Finch\Downloads\minbar_mock'

all_files = os.listdir(min_bar_path)
all_files.sort()
all_files = all_files[-5:]

min_bar_datas = pd.DataFrame()
for file in all_files:
    temp = pd.read_parquet(min_bar_path + '\\' + file + '\\AllSymbols_1min.parquet')
    temp = temp[temp['symbol'].isin(stkcds)]
    min_bar_datas = min_bar_datas.append(temp)

min_bar_datas = min_bar_datas[['symbol', 'datetime', 'turnover']]
min_bar_datas['datetime'] = min_bar_datas.apply(lambda x: str(x['datetime'])[11:], axis=1)
min_bar_datas = min_bar_datas.groupby(['symbol', 'datetime']).agg({'turnover': 'mean'}).reset_index()
min_bar_datas = min_bar_datas[min_bar_datas['datetime'] != '09:25:00'].reset_index(drop=True)

datetime_dir = pd.read_csv(r'C:\PycharmProjects\PorfolioMonitor\cache\timestamp_dir.csv')

datetime_dir = datetime_dir['datetime'].to_dict()
datetime_dir = dict([val, key] for key, val in datetime_dir.items())

min_bar_datas = pd.DataFrame(min_bar_datas).replace(datetime_dir)
min_bar_datas.to_csv(r'C:\PycharmProjects\PorfolioMonitor\historical_market_data\average_turnover.csv',index = False)
