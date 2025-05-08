from __future__ import annotations

from typing import TYPE_CHECKING, List, Union, cast

import numpy as np
import pandas as pd

from ._util import _data_period, _indicator_warmup_nbars

if TYPE_CHECKING:
    from .backtesting import Strategy, Trade


def compute_drawdown_duration_peaks(dd: pd.Series):
    """
    计算回撤周期和最大回撤。

    参数:
        dd (pd.Series): 回撤序列。

    返回:
        tuple: (回撤周期, 最大回撤)
    """
    iloc = np.unique(np.r_[(dd == 0).values.nonzero()[0], len(dd) - 1])
    iloc = pd.Series(iloc, index=dd.index[iloc])
    df = iloc.to_frame('iloc').assign(prev=iloc.shift())
    df = df[df['iloc'] > df['prev'] + 1].astype(np.int64)

    # 如果没有交易，避免后续操作并返回 NaN 序列
    if not len(df):
        return (dd.replace(0, np.nan),) * 2

    df['duration'] = df['iloc'].map(dd.index.__getitem__) - df['prev'].map(dd.index.__getitem__)
    df['peak_dd'] = df.apply(lambda row: dd.iloc[row['prev']:row['iloc'] + 1].max(), axis=1)
    df = df.reindex(dd.index)
    return df['duration'], df['peak_dd']


def geometric_mean(returns: pd.Series) -> float:
    """
    计算几何平均收益。

    参数:
        returns (pd.Series): 收益率序列。

    返回:
        float: 几何平均收益率。
    """
    returns = returns.fillna(0) + 1
    if np.any(returns <= 0):
        return 0
    return np.exp(np.log(returns).sum() / (len(returns) or np.nan)) - 1


def compute_stats(
        trades: Union[List['Trade'], pd.DataFrame],
        equity: np.ndarray,
        ohlc_data: pd.DataFrame,
        strategy_instance: Strategy | None,
        risk_free_rate: float = 0,
) -> pd.Series:
    """
    计算回测统计指标。

    参数:
        trades (Union[List['Trade'], pd.DataFrame]): 交易列表或 DataFrame。
        equity (np.ndarray): 权益曲线数组。
        ohlc_data (pd.DataFrame): OHLC 数据。
        strategy_instance (Strategy | None): 策略实例。
        risk_free_rate (float): 无风险利率（默认为 0）。

    返回:
        pd.Series: 包含所有统计指标的 Series。
    """
    assert -1 < risk_free_rate < 1

    index = ohlc_data.index
    dd = 1 - equity / np.maximum.accumulate(equity)
    dd_dur, dd_peaks = compute_drawdown_duration_peaks(pd.Series(dd, index=index))

    equity_df = pd.DataFrame({
        'Equity': equity,
        'DrawdownPct': dd,
        'DrawdownDuration': dd_dur},
        index=index)

    if isinstance(trades, pd.DataFrame):
        trades_df: pd.DataFrame = trades
        commissions = None  # 不显示佣金
    else:
        # 直接来自 Backtest.run()
        trades_df = pd.DataFrame({
            'Size': [t.size for t in trades],
            'EntryBar': [t.entry_bar for t in trades],
            'ExitBar': [t.exit_bar for t in trades],
            'EntryPrice': [t.entry_price for t in trades],
            'ExitPrice': [t.exit_price for t in trades],
            'SL': [t.sl for t in trades],
            'TP': [t.tp for t in trades],
            'PnL': [t.pl for t in trades],
            'ReturnPct': [t.pl_pct for t in trades],
            'EntryTime': [t.entry_time for t in trades],
            'ExitTime': [t.exit_time for t in trades],
        })
        trades_df['Duration'] = trades_df['ExitTime'] - trades_df['EntryTime']
        trades_df['Tag'] = [t.tag for t in trades]

        # 添加指标值
        if len(trades_df) and strategy_instance:
            for ind in strategy_instance._indicators:
                ind = np.atleast_2d(ind)
                for i, values in enumerate(ind):  # 多维指标
                    suffix = f'_{i}' if len(ind) > 1 else ''
                    trades_df[f'Entry_{ind.name}{suffix}'] = values[trades_df['EntryBar'].values]
                    trades_df[f'Exit_{ind.name}{suffix}'] = values[trades_df['ExitBar'].values]

        commissions = sum(t._commissions for t in trades)
    del trades

    pl = trades_df['PnL']
    returns = trades_df['ReturnPct']
    durations = trades_df['Duration']

    def _round_timedelta(value, _period=_data_period(index)):
        if not isinstance(value, pd.Timedelta):
            return value
        resolution = getattr(_period, 'resolution_string', None) or _period.resolution
        return value.ceil(resolution)

    s = pd.Series(dtype=object)
    s.loc['Start'] = index[0]
    s.loc['End'] = index[-1]
    s.loc['Duration'] = s.End - s.Start

    have_position = np.repeat(0, len(index))
    for t in trades_df.itertuples(index=False):
        have_position[t.EntryBar:t.ExitBar + 1] = 1

    s.loc['Exposure Time [%]'] = have_position.mean() * 100  # 在“n bars”时间中的暴露时间比例
    s.loc['Equity Final [$]'] = equity[-1]
    s.loc['Equity Peak [$]'] = equity.max()
    if commissions:
        s.loc['Commissions [$]'] = commissions
    s.loc['Return [%]'] = (equity[-1] - equity[0]) / equity[0] * 100
    first_trading_bar = _indicator_warmup_nbars(strategy_instance)
    c = ohlc_data.Close.values
    s.loc['Buy & Hold Return [%]'] = (c[-1] - c[first_trading_bar]) / c[first_trading_bar] * 100  # 长期持有回报

    gmean_day_return: float = 0
    day_returns = np.array(np.nan)
    annual_trading_days = np.nan
    is_datetime_index = isinstance(index, pd.DatetimeIndex)
    if is_datetime_index:
        freq_days = cast(pd.Timedelta, _data_period(index)).days
        have_weekends = index.dayofweek.to_series().between(5, 6).mean() > 2 / 7 * .6
        annual_trading_days = (
            52 if freq_days == 7 else
            12 if freq_days == 31 else
            1 if freq_days == 365 else
            (365 if have_weekends else 252))
        freq = {7: 'W', 31: 'ME', 365: 'YE'}.get(freq_days, 'D')
        day_returns = equity_df['Equity'].resample(freq).last().dropna().pct_change()
        gmean_day_return = geometric_mean(day_returns)

    # 年化收益和风险指标基于复利假设计算。详见：https://dx.doi.org/10.2139/ssrn.3054517
    annualized_return = (1 + gmean_day_return) ** annual_trading_days - 1
    s.loc['Return (Ann.) [%]'] = annualized_return * 100
    s.loc['Volatility (Ann.) [%]'] = np.sqrt(
        (day_returns.var(ddof=int(bool(day_returns.shape))) + (1 + gmean_day_return) ** 2) ** annual_trading_days - (
                    1 + gmean_day_return) ** (2 * annual_trading_days)) * 100  # noqa: E501
    if is_datetime_index:
        time_in_years = (s.loc['Duration'].days + s.loc['Duration'].seconds / 86400) / annual_trading_days
        s.loc['CAGR [%]'] = ((s.loc['Equity Final [$]'] / equity[0]) ** (
                    1 / time_in_years) - 1) * 100 if time_in_years else np.nan  # noqa: E501

    # 我们的 Sharpe Ratio 和 `empyrical.sharpe_ratio()` 不匹配，因为它们使用算术平均收益和简单标准差
    s.loc['Sharpe Ratio'] = (s.loc['Return (Ann.) [%]'] - risk_free_rate * 100) / (
                s.loc['Volatility (Ann.) [%]'] or np.nan)  # noqa: E501
    # 我们的 Sortino Ratio 和 `empyrical.sortino_ratio()` 不匹配，因为它们使用算术平均收益
    with np.errstate(divide='ignore'):
        s.loc['Sortino Ratio'] = (annualized_return - risk_free_rate) / (
                    np.sqrt(np.mean(day_returns.clip(-np.inf, 0) ** 2)) * np.sqrt(annual_trading_days))  # noqa: E501
    max_dd = -np.nan_to_num(dd.max())
    s.loc['Calmar Ratio'] = annualized_return / (-max_dd or np.nan)
    equity_log_returns = np.log(equity[1:] / equity[:-1])
    market_log_returns = np.log(c[1:] / c[:-1])
    beta = np.nan
    if len(equity_log_returns) > 1 and len(market_log_returns) > 1:
        cov_matrix = np.cov(equity_log_returns, market_log_returns)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
    # Jensen CAPM Alpha: 当 beta 为负且 B&H Return 很大时可能为正
    s.loc['Alpha [%]'] = s.loc['Return [%]'] - risk_free_rate * 100 - beta * (
                s.loc['Buy & Hold Return [%]'] - risk_free_rate * 100)  # noqa: E501
    s.loc['Beta'] = beta
    s.loc['Max. Drawdown [%]'] = max_dd * 100
    s.loc['Avg. Drawdown [%]'] = -dd_peaks.mean() * 100
    s.loc['Max. Drawdown Duration'] = _round_timedelta(dd_dur.max())
    s.loc['Avg. Drawdown Duration'] = _round_timedelta(dd_dur.mean())
    s.loc['# Trades'] = n_trades = len(trades_df)
    win_rate = np.nan if not n_trades else (pl > 0).mean()
    s.loc['Win Rate [%]'] = win_rate * 100
    s.loc['Best Trade [%]'] = returns.max() * 100
    s.loc['Worst Trade [%]'] = returns.min() * 100
    mean_return = geometric_mean(returns)
    s.loc['Avg. Trade [%]'] = mean_return * 100
    s.loc['Max. Trade Duration'] = _round_timedelta(durations.max())
    s.loc['Avg. Trade Duration'] = _round_timedelta(durations.mean())
    s.loc['Profit Factor'] = returns[returns > 0].sum() / (abs(returns[returns < 0].sum()) or np.nan)  # noqa: E501
    s.loc['Expectancy [%]'] = returns.mean() * 100
    s.loc['SQN'] = np.sqrt(n_trades) * pl.mean() / (pl.std() or np.nan)
    s.loc['Kelly Criterion'] = win_rate - (1 - win_rate) / (pl[pl > 0].mean() / -pl[pl < 0].mean())

    s.loc['_strategy'] = strategy_instance
    s.loc['_equity_curve'] = equity_df
    s.loc['_trades'] = trades_df

    s = _Stats(s)
    return s


class _Stats(pd.Series):
    def __repr__(self):
        with pd.option_context(
                'display.max_colwidth', 20,  # 防止由于 _equity 和 _trades 数据帧扩展
                'display.max_rows', len(self),  # 显示完整内容
                'display.precision', 5,  # 足够的精度
        ):
            return super().__repr__()


def dummy_stats():
    """
    返回一个虚拟的统计结果用于测试。
    """
    from .backtesting import Trade, _Broker
    index = pd.DatetimeIndex(['2025'])
    data = pd.DataFrame({col: [np.nan] for col in ('Close',)}, index=index)
    trade = Trade(_Broker(data=data, cash=10000, spread=.01, commission=.01, margin=.1,
                          trade_on_close=True, hedging=True, exclusive_orders=False, index=index),
                  1, 1, 0, None)
    trade._replace(exit_price=1, exit_bar=0)
    trade._commissions = np.nan
    return compute_stats([trade], np.r_[[np.nan]], data, None, 0)
