# coding=utf-8
from __future__ import print_function, absolute_import, unicode_literals
import numpy as np
from gm.api import *
from pandas import DataFrame

'''
本策略每隔1个月定时触发,根据Fama-French三因子模型对每只股票进行回归，得到其alpha值。
假设Fama-French三因子模型可以完全解释市场，则alpha为负表明市场低估该股，因此应该买入。
策略思路：
计算市场收益率、个股的账面市值比和市值,并对后两个进行了分类,
根据分类得到的组合分别计算其市值加权收益率、SMB和HML. 
对各个股票进行回归(假设无风险收益率等于0)得到alpha值.
选取alpha值小于0并为最小的10只股票进入标的池
平掉不在标的池的股票并等权买入在标的池的股票
回测数据:SHSE.000300的成份股
回测时间:2017-07-01 08:00:00到2017-10-01 16:00:00
'''
# context 上下文，全局变量可存储在这里
def init(context):
    # 每月第一个交易日的09:40 定时执行algo任务（仿真和实盘时不支持该频率）
    # algo执行定时任务函数，只能传context参数
    # date_rule执行频率，目前暂时支持1d、1w、1m，其中1w、1m仅用于回测，实时模式1d以上的频率，需要在algo判断日期
    # time_rule执行时间， 注意多个定时任务设置同一个时间点，前面的定时任务会被后面的覆盖
    schedule(schedule_func=algo, date_rule='1m', time_rule='09:40:00')

    # 数据滑窗
    context.date = 20

    # 设置开仓的最大资金量
    context.ratio = 0.8

    # 账面市值比的大/中/小分类
    context.BM_BIG = 3.0
    context.BM_MID = 2.0
    context.BM_SMA = 1.0

    # 市值大/小分类
    context.MV_BIG = 2.0
    context.MV_SMA = 1.0

# 计算市值加权的收益率的函数,MV为市值的分类对应的组别,BM为账目市值比的分类对应的组别
def market_value_weighted(stocks, MV, BM):
    select = stocks[(stocks['NEGOTIABLEMV'] == MV) & (stocks['BM'] == BM)] # 选出市值为MV，账目市值比为BM的所有股票数据
    market_value = select['mv'].values     # 对应组的全部市值数据
    mv_total = np.sum(market_value)        # 市值求和
    mv_weighted = [mv / mv_total for mv in market_value]   # 市值加权的权重
    stock_return = select['return'].values

    # 返回市值加权的收益率的和
    return_total = []
    for i in range(len(mv_weighted)):
        return_total.append(mv_weighted[i] * stock_return[i])
    return_total = np.sum(return_total)
    return return_total


def algo(context):
    # 获取上一个交易日的日期
    last_day = get_previous_trading_date(exchange='SHSE', date=context.now)

    # 获取沪深300成份股
    context.stock300 = get_history_constituents(index='SHSE.000300', start_date=last_day,
                                                end_date=last_day)[0]['constituents'].keys()

    # 获取当天有交易的股票
    not_suspended = get_history_instruments(symbols=context.stock300, start_date=last_day, end_date=last_day)
    not_suspended = [item['symbol'] for item in not_suspended
                     if not item['is_suspended'] and item['symbol']!= 'SHSE.601313']  # 601313 为退市股票

    # 查询基本面数据
    fin = get_fundamentals(table='trading_derivative_indicator', symbols=not_suspended,
                           start_date=last_day, end_date=last_day,fields='PB,NEGOTIABLEMV', df=True)  # 获取PB：（市净率(PB)）和NEGOTIABLEMV：（流通市值）

    # 计算账面市值比,为P/B的倒数
    fin['PB'] = (fin['PB'] ** -1)

    # 计算市值的50%的分位点,用于后面的分类
    size_gate = fin['NEGOTIABLEMV'].quantile(0.50)

    # 计算账面市值比的30%和70%分位点,用于后面的分类
    bm_gate = [fin['PB'].quantile(0.30), fin['PB'].quantile(0.70)]
    fin.index = fin.symbol

    # 设置存放股票收益率的list
    x_return = []

    # 对未停牌的股票进行处理
    for symbol in not_suspended:
        # 计算收益率，存放到x_return里面
        close = history_n(symbol=symbol, frequency='1d', count=context.date + 1, end_time=last_day, fields='close',
                          skip_suspended=True, fill_missing='Last', adjust=ADJUST_PREV, df=True)['close'].values
        stock_return = close[-1] / close[0] - 1
        pb = fin['PB'][symbol]
        market_value = fin['NEGOTIABLEMV'][symbol]

        # 获取[股票代码， 股票收益率, 账面市值比的分类, 市值的分类, 流通市值]
        # 其中账面市值比的分类为：大（3）、中（2）、小（1）
        # 流通市值的分类：大（2）、小（1）
        if pb < bm_gate[0]:
            if market_value < size_gate:
                label = [symbol, stock_return, context.BM_SMA, context.MV_SMA, market_value]
            else:
                label = [symbol, stock_return, context.BM_SMA, context.MV_BIG, market_value]
        elif pb < bm_gate[1]:
            if market_value < size_gate:
                label = [symbol, stock_return, context.BM_MID, context.MV_SMA, market_value]
            else:
                label = [symbol, stock_return, context.BM_MID, context.MV_BIG, market_value]
        elif market_value < size_gate:
            label = [symbol, stock_return, context.BM_BIG, context.MV_SMA, market_value]
        else:
            label = [symbol, stock_return, context.BM_BIG, context.MV_BIG, market_value]
        if len(x_return) == 0:
            x_return = label
        else:
            x_return = np.vstack([x_return, label])

    # 将股票代码、 股票收益率、 账面市值比的分类、 市值的分类、 流通市值存为数据表
    stocks = DataFrame(data=x_return, columns=['symbol', 'return', 'BM', 'NEGOTIABLEMV', 'mv'])
    stocks.index = stocks.symbol
    columns = ['return', 'BM', 'NEGOTIABLEMV', 'mv']

    for column in columns:
        stocks[column] = stocks[column].astype(np.float64)

    # 计算SMB.HML和市场收益率（市值加权法）
    smb_s = (market_value_weighted(stocks, context.MV_SMA, context.BM_SMA) +
             market_value_weighted(stocks, context.MV_SMA, context.BM_MID) +
             market_value_weighted(stocks, context.MV_SMA, context.BM_BIG)) / 3

    # 获取大市值组合的市值加权组合收益率
    smb_b = (market_value_weighted(stocks, context.MV_BIG, context.BM_SMA) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_MID) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_BIG)) / 3
    smb = smb_s - smb_b

    # 获取大账面市值比组合的市值加权组合收益率
    hml_b = (market_value_weighted(stocks, context.MV_SMA, 3) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_BIG)) / 2

    # 获取小账面市值比组合的市值加权组合收益率
    hml_s = (market_value_weighted(stocks, context.MV_SMA, context.BM_SMA) +
             market_value_weighted(stocks, context.MV_BIG, context.BM_SMA)) / 2
    hml = hml_b - hml_s

    # 获取市场收益率
    close = history_n(symbol='SHSE.000300', frequency='1d', count=context.date + 1,
                      end_time=last_day, fields='close', skip_suspended=True,
                      fill_missing='Last', adjust=ADJUST_PREV, df=True)['close'].values
    market_return = close[-1] / close[0] - 1
    coff_pool = []

    # 对每只股票进行回归获取其alpha值
    for stock in stocks.index:
        x_value = np.array([[market_return], [smb], [hml], [1.0]])
        y_value = np.array([stocks['return'][stock]])
        # OLS估计系数
        coff = np.linalg.lstsq(x_value.T, y_value)[0][3]
        coff_pool.append(coff)

    # 获取alpha最小并且小于0的10只的股票进行操作(若少于10只则全部买入)
    stocks['alpha'] = coff_pool
    stocks = stocks[stocks.alpha < 0].sort_values(by='alpha').head(10)
    symbols_pool = stocks.index.tolist()
    positions = context.account().positions()

    # 平不在标的池的股票
    for position in positions:
        symbol = position['symbol']
        if symbol not in symbols_pool:
            order_target_percent(symbol=symbol, percent=0, order_type=OrderType_Market,
                                 position_side=PositionSide_Long)
            print('市价单平不在标的池的', symbol)

    # 获取股票的权重
    percent = context.ratio / len(symbols_pool)

    # 买在标的池中的股票
    for symbol in symbols_pool:
        # 调仓到目标持仓比例（总资产的比例）
        # symbol str 标的代码
        # percent double 期望的最终占总资产比例
        # position_side int 表示将多仓还是空仓调到目标持仓比例，参见持仓方向 PositionSide_Long 多方向
        # order_type int 参见订单类型 OrderType_Market 市价委托
        order_target_percent(symbol=symbol, percent=percent, order_type=OrderType_Market,
                             position_side=PositionSide_Long)
        print(symbol, '以市价单调多仓到仓位', percent)


if __name__ == '__main__':
    '''
    strategy_id策略ID,由系统生成
    filename文件名,请与本文件名保持一致
    mode实时模式:MODE_LIVE回测模式:MODE_BACKTEST
    token绑定计算机的ID,可在系统设置-密钥管理中生成
    backtest_start_time回测开始时间
    backtest_end_time回测结束时间
    backtest_adjust股票复权方式不复权:ADJUST_NONE前复权:ADJUST_PREV后复权:ADJUST_POST
    backtest_initial_cash回测初始资金
    backtest_commission_ratio回测佣金比例
    backtest_slippage_ratio回测滑点比例
    '''
    run(strategy_id='5ee6050a-6edc-11ec-bf20-00ff38297386',
        filename='main.py',
        mode=MODE_BACKTEST,
        token='189745a1f225972a97df1653091411273f871573',
        backtest_start_time='2021-01-06 08:00:00',
        backtest_end_time='2022-01-06 16:00:00',
        backtest_adjust=ADJUST_PREV,
        backtest_initial_cash=10000000,
        backtest_commission_ratio=0.0001,
        backtest_slippage_ratio=0.0001)

