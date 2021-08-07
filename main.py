import os
import time
import warnings
import datetime

import pandas as pd
import requests

from bokeh.models import Axis, HoverTool, Panel, Tabs
from bokeh.models import ColumnDataSource
from bokeh.models import PreText
from bokeh.models import Button
from bokeh.models import CustomJS
from bokeh.models import DatePicker
from bokeh.models import Select
from bokeh.models import FileInput

from bokeh.io import curdoc

from bokeh.layouts import column, row
from bokeh.plotting import figure

warnings.filterwarnings("ignore")


# warnings.warn("hello")

def turnover_format_translate(past_min_turnover):
    past_min_turnover['stkcd'] = past_min_turnover.apply(
        lambda x: str(x['symbol'])[7:9].lower() + str(x['symbol'])[:6], axis=1)
    past_min_turnover = pd.merge(
        left=porfolios_file,
        right=past_min_turnover,
        on='stkcd',
        how='outer')
    past_min_turnover['weighted_turnover'] = past_min_turnover.apply(
        lambda x: float(x['weight']) * float(x['turnover']),
        axis=1)
    past_min_turnover = past_min_turnover.groupby(['flag', 'datetime']).agg(
        {'weighted_turnover': 'sum'}).reset_index()
    past_min_turnover.rename(columns={'datetime': 'time_stamp'}, inplace=True)
    # delete row which weighted_turnover equals zero
    past_min_turnover = past_min_turnover[past_min_turnover['weighted_turnover'] != 0]
    return past_min_turnover


path = os.path
print("Running path:" + str(path))

now = datetime.datetime.now()
date = str(now.year) + '_' + str(now.month) + '_' + str(now.day)
print("Today:" + date)

data_file_path = os.path.abspath(
    '.') + '\\PorfolioMonitor\\historical_market_data\\daily_data' + '_' + date + '.pickle'

print('Daily market data path:' + data_file_path)
past_min_turnover_file_path = os.path.abspath(
    '.') + '\\PorfolioMonitor\\historical_market_data\\past_five_days_turnover' + '_' + date + '.pickle'
print("Past min turnover file path:" + past_min_turnover_file_path)
datetime_dir = pd.read_csv(os.path.abspath('.') +
                           '\\PorfolioMonitor\\cache\\timestamp_dir.csv')

min_data_path = r'C:\Users\Finch\Downloads\minbar_mock'
print("min data file path:" + min_data_path)
porfolio_files_path = os.path.abspath(
    '.') + '\\PorfolioMonitor\\porfolio_files'
porfolios = os.listdir(porfolio_files_path)
porfolios_file = pd.DataFrame()
for i in porfolios:
    temp_df = pd.read_excel(porfolio_files_path + '\\' + i)
    temp_df['flag'] = str(i.split('.')[0])
    temp_df['base'] = str(i[:2])
    porfolios_file = porfolios_file.append(temp_df)


def calculate_past_five_day_min_turnover(porfolios_file, min_bar_path):
    # datetime_dir = pd.read_csv(r'C:\PycharmProjects\PorfolioMonitor\cache\timestamp_dir.csv')
    global datetime_dir
    # 读入每只股票的min bar信息
    stkcds = pd.DataFrame(porfolios_file['stkcd'])
    stkcds['stkcd'] = stkcds.apply(lambda x: str(
        x['stkcd'])[2:] + '.' + str(x['stkcd'])[:2].upper(), axis=1)
    stkcds = stkcds['stkcd'].values

    # minBar database path

    all_files = sorted(os.listdir(min_bar_path))
    all_files = all_files[-5:]

    min_bar_datas = pd.DataFrame()
    for file in all_files:
        temp = pd.read_parquet(
            min_bar_path +
            '\\' +
            file +
            '\\AllSymbols_1min.parquet')
        temp = temp[temp['symbol'].isin(stkcds)]
        min_bar_datas = min_bar_datas.append(temp)

    min_bar_datas = min_bar_datas[['symbol', 'datetime', 'turnover']]
    min_bar_datas['datetime'] = min_bar_datas.apply(
        lambda x: str(x['datetime'])[11:], axis=1)
    min_bar_datas = min_bar_datas.groupby(['symbol', 'datetime']).agg({
        'turnover': 'mean'}).reset_index()
    min_bar_datas = min_bar_datas[min_bar_datas['datetime']
                                  != '09:25:00'].reset_index(drop=True)

    datetime_dir = datetime_dir['datetime'].to_dict()
    datetime_dir = dict([val, key] for key, val in datetime_dir.items())

    min_bar_datas = pd.DataFrame(min_bar_datas).replace(datetime_dir)
    min_bar_datas.to_csv(past_min_turnover_file_path, index=False)
    return min_bar_datas


if not os.path.exists(past_min_turnover_file_path):
    # TODO:修改路径
    past_min_turnover = calculate_past_five_day_min_turnover(
        porfolios_file, min_data_path)
else:
    past_min_turnover = pd.read_csv(past_min_turnover_file_path)

past_min_turnover = turnover_format_translate(past_min_turnover)
past_min_turnover['flag'] = past_min_turnover.apply(
    lambda x: str(x['flag']).lower(), axis=1)


def initialize_market_information(time_stamp, data_file):
    porfolios = porfolios_file

    url = 'http://hq.sinajs.cn/list='
    for index, row in porfolios.iterrows():
        url = url + str(row['stkcd']) + ','

    r = requests.get(url)
    stk_infos = pd.DataFrame(r.text.split('\n'), columns=['all_info'])
    stk_infos = stk_infos[:-1]

    stk_infos['stkcd'] = stk_infos.apply(
        lambda x: x['all_info'][11:19], axis=1)
    stk_infos['metrics'] = stk_infos.apply(
        lambda x: x['all_info'][21:-2], axis=1)
    stk_infos['close_price'] = stk_infos.apply(
        lambda x: x['metrics'].split(',')[2], axis=1)
    stk_infos['now_price'] = stk_infos.apply(
        lambda x: x['metrics'].split(',')[3], axis=1)
    stk_infos['time'] = stk_infos.apply(
        lambda x: x['metrics'].split(',')[31], axis=1)
    stk_infos['change_rate'] = stk_infos.apply(
        lambda x: (
                          (float(
                              x['now_price']) -
                           float(
                               x['close_price'])) /
                          float(
                              x['close_price'])) *
                  100,
        axis=1)
    stk_infos['current_turnover'] = stk_infos.apply(
        lambda x: str(x['all_info']).split(',')[9], axis=1)

    stk_infos = stk_infos[['stkcd', 'change_rate', 'current_turnover']]

    porfolios = pd.merge(
        left=porfolios,
        right=stk_infos,
        on='stkcd').drop_duplicates()
    porfolios['time_stamp'] = time_stamp

    # 沪深300
    hs300index_url = requests.get('http://hq.sinajs.cn/list=s_sh000300')
    hs300index_rate = float(str(hs300index_url.text).split(',')[-3])

    # 中证500
    zz500index_url = requests.get('http://hq.sinajs.cn/list=s_sh000905')
    zz500index_rate = float(str(zz500index_url.text).split(',')[-3])

    final_porfolio_hs = porfolios[porfolios['base'] == 'HS']
    final_porfolio_zz = porfolios[porfolios['base'] == 'ZZ']

    final_porfolio_hs['excess_rate'] = final_porfolio_hs.apply(
        lambda x: (float(x['change_rate']) - float(hs300index_rate)), axis=1)
    final_porfolio_zz['excess_rate'] = final_porfolio_zz.apply(
        lambda x: (float(x['change_rate']) - float(zz500index_rate)), axis=1)

    final_porfolio = final_porfolio_hs.append(final_porfolio_zz)

    final_porfolio.to_pickle(data_file)


if not os.path.exists(data_file_path):
    initialize_market_information(1, data_file_path)


def update_market_information(stamp, data_file):
    porfolios = porfolios_file

    url = 'http://hq.sinajs.cn/list='
    for index, row in porfolios.iterrows():
        url = url + str(row['stkcd']) + ','

    r = requests.get(url)
    stk_infos = pd.DataFrame(r.text.split('\n'), columns=['all_info'])
    stk_infos = stk_infos[:-1]

    stk_infos['stkcd'] = stk_infos.apply(
        lambda x: x['all_info'][11:19], axis=1)
    stk_infos['metrics'] = stk_infos.apply(
        lambda x: x['all_info'][21:-2], axis=1)
    stk_infos['close_price'] = stk_infos.apply(
        lambda x: x['metrics'].split(',')[2], axis=1)
    stk_infos['now_price'] = stk_infos.apply(
        lambda x: x['metrics'].split(',')[3], axis=1)
    stk_infos['time'] = stk_infos.apply(
        lambda x: x['metrics'].split(',')[31], axis=1)
    stk_infos['change_rate'] = stk_infos.apply(
        lambda x: (
                          (float(
                              x['now_price']) -
                           float(
                               x['close_price'])) /
                          float(
                              x['close_price'])) *
                  100,
        axis=1)

    stk_infos['current_turnover'] = stk_infos.apply(
        lambda x: str(x['all_info']).split(',')[9], axis=1)

    stk_infos = stk_infos[['stkcd', 'change_rate', 'current_turnover']]

    porfolios = pd.merge(
        left=porfolios,
        right=stk_infos,
        on='stkcd').drop_duplicates()
    porfolios['time_stamp'] = stamp

    # 沪深300
    hs300index_url = requests.get('http://hq.sinajs.cn/list=s_sh000300')
    hs300index_rate = float(str(hs300index_url.text).split(',')[-3])

    # 中证500
    zz500index_url = requests.get('http://hq.sinajs.cn/list=s_sh000905')
    zz500index_rate = float(str(zz500index_url.text).split(',')[-3])

    final_porfolio_hs = porfolios[porfolios['base'] == 'HS']
    final_porfolio_zz = porfolios[porfolios['base'] == 'ZZ']

    final_porfolio_hs['excess_rate'] = final_porfolio_hs.apply(
        lambda x: (float(x['change_rate']) - float(hs300index_rate)), axis=1)
    final_porfolio_zz['excess_rate'] = final_porfolio_zz.apply(
        lambda x: (float(x['change_rate']) - float(zz500index_rate)), axis=1)

    final_porfolio = final_porfolio_hs.append(final_porfolio_zz)

    result = pd.read_pickle(data_file)

    # TODO: test setting
    # result = result.append(final_porfolio)
    # result.to_pickle(data_file)
    # return result

    now_time = datetime.datetime.now()
    t_0 = datetime.time(9, 30, 00)
    t_1 = datetime.time(11, 30, 00)
    t_2 = datetime.time(13, 30, 00)
    t_4 = datetime.time(15, 00, 00)
    global time_stamp
    if now_time.time() < t_0:
        time_stamp = time_stamp - 1
        return result

    if now_time.time() > t_4:
        time_stamp = time_stamp - 1
        return result

    if ((t_1 < now_time.time()) & (now_time.time() < t_2)):
        time_stamp = time_stamp - 1
        return result
    else:
        result = result.append(final_porfolio)
        result.to_pickle(data_file)
        return result


def calculate_weighted_price(stk_infos, flag):
    # flag = 'HS300'
    stk_infos = stk_infos[stk_infos['flag'] == flag]

    stk_infos['weighted_rate'] = stk_infos.apply(
        lambda x: float(x['change_rate']) * float(x['weight']), axis=1)

    stk_infos['weighted_excess_rate'] = stk_infos.apply(
        lambda x: float(x['excess_rate']) * float(x['weight']), axis=1)

    stk_infos['weighted_current_turnover'] = stk_infos.apply(
        lambda x: float(x['current_turnover']) * float(x['weight']), axis=1)

    weighted_ratio = stk_infos.groupby(by='time_stamp').agg(
        {'weighted_rate': 'sum', 'weighted_excess_rate': 'sum'}).reset_index()

    ind_ratio = stk_infos.groupby(by=['time_stamp', 'ind']).agg(
        {'weighted_rate': 'sum', 'weighted_excess_rate': 'sum'}).reset_index()

    weighted_turnover = stk_infos.groupby(by='time_stamp').agg(
        {'weighted_current_turnover': 'sum'}).reset_index()

    stk_infos = stk_infos[['stkcd', 'time_stamp', 'current_turnover']]

    return [weighted_ratio, ind_ratio, weighted_turnover]


TOOLTIPS = """
    <div>
        <div>
            <span style="font-size: 13px; color: black;"> $name </span>
        </div>
        <div>
            <span style="font-size: 10px; color: #696;">timeStamp: $x{0.} min</span>
        </div>
        <div>
            <span style="font-size: 10px; color: #696;">returnRatio: $y{0.000} %</span>
        </div>
    </div>
"""
frequency = 60000

colors = list(pd.read_csv(
    r'C:\PycharmProjects\PorfolioMonitor\cache\colors.csv')['color'].tolist())

porfolios_name = ["HS300", "HS300LV", "ZZ500VG", "ZZ500V", "ZZ500G"]

color_index = 0

board1 = PreText(text='上次更新时间:', width=500)

current_time = PreText(text='等待中', width=500)
frequency = PreText(text='更新频率:' + str(frequency / 1000) + 's', width=500)

tools = 'pan,wheel_zoom,xbox_select,reset'

# source = ColumnDataSource(df)
daily_data = pd.read_pickle(data_file_path)

time_stamp = daily_data['time_stamp'].max()

hs300_dfs = (calculate_weighted_price(daily_data, 'HS300'))
hs300lv_dfs = (calculate_weighted_price(daily_data, 'HS300LV'))
zz500vg_dfs = (calculate_weighted_price(daily_data, 'ZZ500VG'))
zz500v_dfs = (calculate_weighted_price(daily_data, 'ZZ500V'))
zz500g_dfs = (calculate_weighted_price(daily_data, 'ZZ500G'))


def calculate_delta_turnover(porfolio_turnover, flag, past_min_turnover):
    porfolio_turnover['delta_weighted_current_turnover'] = porfolio_turnover['weighted_current_turnover'] - \
                                                           porfolio_turnover['weighted_current_turnover'].shift(1)
    porfolio_turnover = porfolio_turnover[~pd.isna(
        porfolio_turnover['delta_weighted_current_turnover'])]

    porfolio_turnover = porfolio_turnover[porfolio_turnover['time_stamp'] != 1]

    porfolio_turnover = pd.merge(left=porfolio_turnover,
                                 right=past_min_turnover[past_min_turnover['flag'] == flag],
                                 on='time_stamp')

    if len(porfolio_turnover) == 0:
        return pd.DataFrame(columns=['time_stamp', 'turnover_ratio'])

    porfolio_turnover['turnover_ratio'] = porfolio_turnover.apply(lambda x: float(
        x['delta_weighted_current_turnover']) / float(x['weighted_turnover']), axis=1)
    porfolio_turnover = porfolio_turnover[['time_stamp', 'turnover_ratio']]
    return porfolio_turnover


turnover_graph = figure(
    width=1200,
    height=400,
    tools=tools,
    x_axis_type='linear',
    active_drag="xbox_select",
    x_range=(
        1,
        240),
    tooltips=TOOLTIPS)

hs300_turnover = calculate_delta_turnover(
    hs300_dfs[2], 'hs300', past_min_turnover)
hs300_turnover = ColumnDataSource(hs300_turnover)

hs300lv_turnover = calculate_delta_turnover(
    hs300lv_dfs[2], 'hs300lv', past_min_turnover)
hs300lv_turnover = ColumnDataSource(hs300lv_turnover)

zz500vg_turnover = calculate_delta_turnover(
    zz500vg_dfs[2], 'zz500vg', past_min_turnover)
zz500vg_turnover = ColumnDataSource(zz500vg_turnover)

zz500v_turnover = calculate_delta_turnover(
    zz500v_dfs[2], 'zz500v', past_min_turnover)
zz500v_turnover = ColumnDataSource(zz500v_turnover)

zz500g_turnover = calculate_delta_turnover(
    zz500g_dfs[2], 'zz500g', past_min_turnover)
zz500g_turnover = ColumnDataSource(zz500g_turnover)

turnover_graph.line(
    'time_stamp',
    'turnover_ratio',
    source=hs300_turnover,
    legend_label='hs300',
    line_color='blue',
    name='hs300')
turnover_graph.line(
    'time_stamp',
    'turnover_ratio',
    source=hs300lv_turnover,
    legend_label='hs300lv',
    line_color='yellow',
    name='hs300lv')
turnover_graph.line(
    'time_stamp',
    'turnover_ratio',
    source=zz500vg_turnover,
    legend_label='zz500vg',
    line_color='green',
    name='zz500vg')
turnover_graph.line(
    'time_stamp',
    'turnover_ratio',
    source=zz500v_turnover,
    legend_label='zz500v',
    line_color='black',
    name='zz500v')
turnover_graph.line(
    'time_stamp',
    'turnover_ratio',
    source=zz500g_turnover,
    legend_label='zz500g',
    line_color='purple',
    name='zz500g')
turnover_graph.legend.location = "top_right"
turnover_graph.legend.click_policy = "hide"

hs300 = ColumnDataSource(hs300_dfs[0])
hs300lv = ColumnDataSource(hs300lv_dfs[0])
zz500vg = ColumnDataSource(zz500vg_dfs[0])
zz500v = ColumnDataSource(zz500v_dfs[0])
zz500g = ColumnDataSource(zz500g_dfs[0])

porfolios = {
    'HS300': hs300,
    'HS300LV': hs300lv,
    'ZZ500VG': zz500vg,
    'ZZ500V': zz500v,
    'ZZ500G': zz500g}
porfolios_color = ['blue', 'red', 'yellow', 'green', 'purple']

# 创建组合收益率图像
stk_graph = figure(
    width=1200,
    height=400,
    tools=tools,
    x_axis_type='linear',
    active_drag="xbox_select",
    x_range=(
        1,
        240),
    tooltips=TOOLTIPS)

for key, color in zip(porfolios, porfolios_color):
    stk_graph.line('time_stamp',
                   'weighted_rate',
                   source=porfolios[str(key)],
                   legend_label=str(key),
                   line_color=str(color),
                   name=str(key))

stk_graph.legend.location = "top_right"
stk_graph.legend.click_policy = "hide"

# 创建组合超额收益率图像
stk_excess_graph = figure(
    width=1200,
    height=400,
    tools=tools,
    x_axis_type='linear',
    active_drag="xbox_select",
    x_range=(
        1,
        240),
    tooltips=TOOLTIPS)
for key, color in zip(porfolios, porfolios_color):
    stk_excess_graph.line('time_stamp',
                          'weighted_excess_rate',
                          source=porfolios[str(key)],
                          legend_label=str(key),
                          line_color=str(color),
                          name=str(key))

stk_excess_graph.legend.location = "top_right"
stk_excess_graph.legend.click_policy = "hide"

tab1 = Panel(child=stk_graph, title="组合收益")
tab2 = Panel(child=stk_excess_graph, title="超额收益")
tab3 = Panel(child=turnover_graph, title="分钟交易金额比率")

tabs = (Tabs(tabs=[tab1, tab2, tab3]))

ind_graph_hs300 = figure(
    width=1200,
    height=800,
    tools=tools,
    x_axis_type='linear',
    active_drag="xbox_select",
    x_range=(
        1,
        240),
    tooltips=TOOLTIPS)

hs300_ind = (
    pd.pivot(
        hs300_dfs[1],
        index='time_stamp',
        columns='ind',
        values='weighted_rate'))

k = hs300_ind.columns
hs300_ind = ColumnDataSource(hs300_ind)

color_index = 0
for i in k:
    ind_graph_hs300.line(
        'time_stamp',
        i,
        source=hs300_ind,
        legend_label=str(i),
        line_color=colors[color_index],
        name=str(i))
    color_index = color_index + 1

ind_graph_hs300.legend.location = "top_right"
ind_graph_hs300.legend.click_policy = "hide"

ind_graph_hs300lv = figure(
    width=1200,
    height=800,
    tools=tools,
    x_axis_type='linear',
    active_drag="xbox_select",
    x_range=(
        1,
        240),
    tooltips=TOOLTIPS)

hs300lv_ind = pd.pivot(
    hs300lv_dfs[1],
    index='time_stamp',
    columns='ind',
    values='weighted_rate')
k = hs300lv_ind.columns
hs300lv_ind = ColumnDataSource(hs300lv_ind)
color_index = 0
for i in k:
    ind_graph_hs300lv.line(
        'time_stamp',
        i,
        source=hs300lv_ind,
        legend_label=str(i),
        line_color=colors[color_index],
        name=str(i))
    color_index = color_index + 1
ind_graph_hs300lv.legend.location = "top_right"
ind_graph_hs300lv.legend.click_policy = "hide"

ind_graph_zz500vg = figure(
    width=1200,
    height=800,
    tools=tools,
    x_axis_type='linear',
    active_drag="xbox_select",
    x_range=(
        1,
        240),
    tooltips=TOOLTIPS)

zz500vg_ind = pd.pivot(
    zz500vg_dfs[1],
    index='time_stamp',
    columns='ind',
    values='weighted_rate')

k = zz500vg_ind.columns
zz500vg_ind = ColumnDataSource(zz500vg_ind)
color_index = 0

for i in k:
    ind_graph_zz500vg.line(
        'time_stamp',
        i,
        source=zz500vg_ind,
        legend_label=str(i),
        line_color=colors[color_index],
        name=str(i))
    color_index = color_index + 1
ind_graph_zz500vg.legend.location = "top_right"
ind_graph_zz500vg.legend.click_policy = "hide"

ind_graph_zz500v = figure(
    width=1200,
    height=800,
    tools=tools,
    x_axis_type='linear',
    active_drag="xbox_select",
    x_range=(
        1,
        240),
    tooltips=TOOLTIPS)

zz500v_ind = pd.pivot(
    zz500v_dfs[1],
    index='time_stamp',
    columns='ind',
    values='weighted_rate')
k = zz500v_ind.columns
zz500v_ind = ColumnDataSource(zz500v_ind)
color_index = 0

for i in k:
    ind_graph_zz500v.line(
        'time_stamp',
        i,
        source=zz500v_ind,
        legend_label=str(i),
        line_color=colors[color_index],
        name=str(i))
    color_index = color_index + 1

ind_graph_zz500v.legend.location = "top_right"
ind_graph_zz500v.legend.click_policy = "hide"

ind_graph_zz500g = figure(
    width=1200,
    height=800,
    tools=tools,
    x_axis_type='linear',
    active_drag="xbox_select",
    x_range=(
        1,
        240),
    tooltips=TOOLTIPS)

zz500g_ind = pd.pivot(
    zz500g_dfs[1],
    index='time_stamp',
    columns='ind',
    values='weighted_rate')
k = zz500g_ind.columns
zz500g_ind = ColumnDataSource(zz500g_ind)
color_index = 0

for i in k:
    ind_graph_zz500g.line(
        'time_stamp',
        i,
        source=zz500g_ind,
        legend_label=str(i),
        line_color=colors[color_index],
        name=str(i))
    color_index = color_index + 1

ind_graph_zz500g.legend.location = "top_right"
ind_graph_zz500g.legend.click_policy = "hide"

tab4 = Panel(child=ind_graph_hs300, title="HS300")
tab5 = Panel(child=ind_graph_hs300lv, title="HS300LV")
tab6 = Panel(child=ind_graph_zz500vg, title="ZZ500VG")
tab7 = Panel(child=ind_graph_zz500v, title="ZZ500V")
tab8 = Panel(child=ind_graph_zz500g, title="ZZ500G")

tabs2 = (Tabs(tabs=[tab4, tab5, tab6, tab7, tab8]))


def update():
    global time_stamp
    time_stamp = time_stamp + 1
    current_time.text = str(datetime.datetime.now())

    result = update_market_information(time_stamp, data_file_path)

    hs300_temp = calculate_weighted_price(result, 'HS300')
    hs300lv_temp = calculate_weighted_price(result, 'HS300LV')
    zz500vg_temp = calculate_weighted_price(result, 'ZZ500VG')
    zz500v_temp = calculate_weighted_price(result, 'ZZ500V')
    zz500g_temp = calculate_weighted_price(result, 'ZZ500G')

    hs300.data = hs300_temp[0]
    hs300lv.data = hs300lv_temp[0]
    zz500vg.data = zz500vg_temp[0]
    zz500v.data = zz500v_temp[0]
    zz500g.data = zz500g_temp[0]

    hs300_ind.data = pd.pivot(
        hs300_temp[1],
        index='time_stamp',
        columns='ind',
        values='weighted_rate')
    hs300lv_ind.data = pd.pivot(
        hs300lv_temp[1],
        index='time_stamp',
        columns='ind',
        values='weighted_rate')
    zz500vg_ind.data = pd.pivot(
        zz500vg_temp[1],
        index='time_stamp',
        columns='ind',
        values='weighted_rate')
    zz500v_ind.data = pd.pivot(
        zz500v_temp[1],
        index='time_stamp',
        columns='ind',
        values='weighted_rate')
    zz500g_ind.data = pd.pivot(
        zz500g_temp[1],
        index='time_stamp',
        columns='ind',
        values='weighted_rate')

    hs300_turnover_temp = calculate_delta_turnover(
        hs300_temp[2], 'hs300', past_min_turnover)
    hs300_turnover.data = hs300_turnover_temp

    hs300lv_turnover_temp = calculate_delta_turnover(
        hs300lv_temp[2], 'hs300lv', past_min_turnover)
    hs300lv_turnover.data = hs300lv_turnover_temp

    zz500vg_turnover_temp = calculate_delta_turnover(
        zz500vg_temp[2], 'zz500vg', past_min_turnover)
    zz500vg_turnover.data = zz500vg_turnover_temp

    zz500v_turnover_temp = calculate_delta_turnover(
        zz500v_temp[2], 'zz500v', past_min_turnover)
    zz500v_turnover.data = zz500v_turnover_temp

    zz500g_turnover_temp = calculate_delta_turnover(
        zz500g_temp[2], 'zz500g', past_min_turnover)
    zz500g_turnover.data = zz500g_turnover_temp


row1 = row(board1, current_time, frequency)
row2 = row(tabs)

ind_board = PreText(text='各组合的行业收益率', width=500)
layout = column(row1, row2, ind_board, tabs2)
curdoc().add_root(layout)
curdoc().title = "Porfolio Monitor"
# curdoc().theme = 'dark_minimal'
# TODO: do not forget to revise update frequency
curdoc().add_periodic_callback(update, 55000)
