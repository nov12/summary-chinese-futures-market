import re
import time
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from tqsdk import TqApi, TqAuth


class TqdataClient:
    """
    Client for querying history data from Tianqin.
    """

    def __init__(self, username: str, password: str):
        self._username = username
        self._password = password
        self.inited = False

        self.only_symbols: dict[list] = {}
        self.current_symbol: dict[list] = {}
        self.daily_data: dict[dict[pd.DataFrame]] = {}

    def login(self) -> bool:
        """"""
        if self.inited:
            return True
        else:
            # self.extremes
            self.api = None

        if not self._username or not self._password:
            return False

        try:
            self.api = TqApi(auth=TqAuth(
                self._username, self._password))

        except Exception as e:
            print(e)
            return False

        self.inited = True
        return True

    def query_contracts(self):
        if not self.inited:
            return None

        self.only_symbols.clear()
        self.current_symbol.clear()

        # 获得全部合约
        symbols = list(self.api.query_quotes(
            ins_class="FUTURE", expired=False))

        # 将合约放入字典中
        for symbol in symbols:
            key, value = symbol.split('.')
            self.current_symbol.setdefault(key, []).append(value)
            self.only_symbols.setdefault(key, set()).add(
                ''.join(filter(str.isalpha, value)))

        # 将合约排序
        for key in self.only_symbols:
            self.only_symbols[key] = sorted(list(self.only_symbols[key]))
            self.current_symbol[key].sort()

    def reconnect(self):
        print(f"{time.asctime()[4:-5]} - 正在重新连接……")
        if self.api:
            self.api.close()
        self.inited = False
        self.login()

    def query_all_history(self):
        if not self.inited:
            return None

        self.daily_data.clear()

        for exchange in self.only_symbols.keys():
            for symbol in self.only_symbols[exchange]:
                contract = f"KQ.i@{exchange}.{symbol}"
                print(f'{time.asctime()[4:-5]} - {contract}获取数据中……')
                df = self.query_history(contract, 86400, 300)

                # 将 datetime 设置为索引
                df.set_index('datetime', inplace=True)

                # 将纳秒时间 index 转换为 datetime index
                # 如果时间不为index，则需要使用Series.dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
                # 参考：https://stackoverflow.com/a/38488959/12846836
                df.index = pd.to_datetime(df.index, unit='ns').tz_localize(
                    'UTC').tz_convert('Asia/Shanghai')

                self.daily_data.setdefault(exchange, {})[symbol] = df
                time.sleep(3)

    def query_history(self, contract: str, interval: int = 86400, length: int = 300) -> pd.DataFrame:
        """
        Query history bar data from Tqdata.
        """
        if not self.inited:
            return None

        # 如果合约不存在，则返回None
        contract_split = contract.split('.')
        if len(contract_split) == 3:
            if contract_split[0] != 'KQ':
                return None
            if contract_split[1].split('@')[1] not in self.current_symbol.keys():
                return None
            if contract_split[2] not in self.only_symbols[contract_split[1].split('@')[1]]:
                return None
        elif len(contract_split) == 2:
            if contract_split[0] not in self.current_symbol.keys():
                return None
            if contract_split[1] not in self.current_symbol[contract_split[0]]:
                return None
        else:
            return None

        # 如果合约存在，且有数据，则返回数据
        df = self.api.get_kline_serial(contract, interval, length)
        print(f"{time.asctime()[4:-5]} - {contract}数据获取完成！共有数据{len(df)}条。")
        self.reconnect()

        return df

    def generate_extreme_dataframe(self, days: list = [90, 180, 270, 365]) -> pd.DataFrame:
        """
        检查峰谷值
        """
        if len(self.daily_data) == 0:
            return

        # 按照日期排序,倒序
        days.sort(reverse=True)

        data = []
        for exchange in self.daily_data.keys():
            for symbol in self.daily_data[exchange].keys():
                df = self.daily_data[exchange][symbol]
                extreme_series = self.get_highest_lowest(df, days)
                proximity_series = self.calculate_proximity(extreme_series)
                series = pd.Series([symbol, 0], index=['symbol', 'direction'])
                series = pd.concat([series, proximity_series, extreme_series])
                data.append(series)

        # 将数据转换为DataFrame
        df_all = pd.DataFrame(data, columns=data[0].index)

        df_all['ration'] = df_all['ration'] * 100
        df_all['extrema_day'] = df_all['extrema_day'].astype(int)

        # 生成方向列
        df_all['direction'] = df_all['extrema_day'].apply(
            lambda x: '多' if x > 0 else '——' if x < 0 else '无')

        # extrema_day列取绝对值
        df_all['extrema_day'] = df_all['extrema_day'].abs()

        # 保留所有float类型的列三位小数
        float_cols = df_all.select_dtypes(include=['float']).columns
        df_all[float_cols] = df_all[float_cols].round(3)

        # 将表头改为中文为生成表格做准备
        df_all.columns = df_all.columns.str.replace('d_', '天内').str.replace('high', '最高')
        df_all.columns = df_all.columns.str.replace('low', '最低').str.replace('today_', '今日')
        df_all.columns = df_all.columns.str.replace('est', '').str.replace('symbol', '合约')
        df_all.columns = df_all.columns.str.replace(
            'ration', '距离%').str.replace('extrema_day', '天数')
        df_all.columns = df_all.columns.str.replace('direction', '方向')

        return df_all

    def get_highest_lowest(self, df: pd.DataFrame, days: list) -> pd.Series:
        """"""
        highest_series = pd.Series()
        lowest_series = pd.Series()

        # 必须设置为相同时区，否则无法比较
        now = pd.Timestamp.now(tz='Asia/Shanghai')

        # 分别计算不同时间段内的最高和最低值，生成pandas.Series。
        for day in days:
            highest = df.loc[now - timedelta(day):]['high'].max()
            highest_series[f'{day}d_highest'] = highest

            lowest = df.loc[now - timedelta(day):]['low'].min()
            lowest_series[f'{day}d_lowest'] = lowest

        # 生成今日最高和最低值的Series
        high = pd.Series([df.iloc[-1]['high']], index=['today_high'])
        low = pd.Series([df.iloc[-1]['low']], index=['today_low'])

        # 将所有Series合并然后返回
        return pd.concat([high, highest_series, low, lowest_series])

    def calculate_proximity(self, series: pd.Series) -> pd.Series:
        """
        计算接近度, 接近度越小, 越接近极值点。
        series 为 get_highest_lowest 返回的 Series, 天数为倒序排列
        """
        proximity = pd.Series()

        length = len(series)

        # 计算接近度
        high_series = series[:length //
                             2].map(lambda x: abs(x - series['today_high']) / series[0])[1:]
        low_series = series[length //
                            2:].map(lambda x: abs(series['today_low'] - x) / series[0])[1:]

        # 获取到接近度的最大索引
        np.where(low_series == low_series.min())[0][-1]
        np.where(high_series == high_series.min())[0][-1]

        # 寻找最近的极值点天数与接近比例
        if low_series.min() > 0.05 and high_series.min() > 0.05:
            proximity['extrema_day'] = 0
            proximity['ration'] = 0
        elif low_series.min() < high_series.min():

            # 获取最小值的索引，series 中天数为倒序排列，所以取第一个。
            index = np.where(low_series == low_series.min())[0][0]

            proximity['extrema_day'] = -1 * int(low_series.index[index].split('d')[0])
            proximity['ration'] = low_series.min()
        else:

            # 获取最小值的索引，series 中天数为倒序排列，所以取第一个。
            index = np.where(high_series == high_series.min())[0][0]

            proximity['extrema_day'] = int(high_series.index[index].split('d')[0])
            proximity['ration'] = high_series.min()

        return proximity

    def get_dominant(self, symbol) -> str:
        """
        查询当前合约的主力合约。
        """
        if not self.inited:
            self.init()

        if '.' in symbol:
            sb, exchange = symbol.split('.')
        else:
            sb = symbol
            exchange = None

        underlying_symbol = re.sub(r'\d', '', sb)

        if not underlying_symbol.isalpha():
            return None

        try:
            dominant_symbol = self.api.query_cont_quotes(
                product_id=underlying_symbol)[0]
            # 如果没有查到则重新改变 underlying_symbol 大小写
        except IndexError:
            if underlying_symbol.islower():
                underlying_symbol = underlying_symbol.upper()
            else:
                underlying_symbol = underlying_symbol.lower()
            dominant_symbol = self.api.query_cont_quotes(
                product_id=underlying_symbol)[0]

        d_exchange, d_symbol = dominant_symbol.split('.')

        if len(dominant_symbol) < 1:
            print("Tqdata Debug 01:", dominant_symbol,
                  re.sub(r'\D', '', dominant_symbol))
            return None

        '''
        dominant_symbol = underlying_symbol + \
            re.sub(r'\D', '', dominant_symbol)
        if self.contract_info[underlying_symbol.upper()]['exchange'] in ['CZCE']:
            if dominant_symbol[1].isdigit():
                dominant_symbol = dominant_symbol[0] + dominant_symbol[2:]
            else:
                dominant_symbol = dominant_symbol[:2] + dominant_symbol[3:]

        if self.contract_info[underlying_symbol.upper()]['exchange'] in ['CZCE', 'CFFEX']:
            dominant_symbol = dominant_symbol.upper()
        else:
            dominant_symbol = dominant_symbol.lower()
        '''

        if exchange is None:
            return d_symbol
        else:
            return f'{d_symbol}.{d_exchange}'

    def is_trading(self, trade_date: datetime = None) -> bool:
        if not self.inited:
            self.init()
        if trade_date is None:
            trade_date = date.today()
        calendar = self.api.get_trading_calendar(trade_date, trade_date)
        return calendar.trading.bool()
