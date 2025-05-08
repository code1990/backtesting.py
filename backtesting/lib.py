"""
collection of common building blocks, helper auxiliary functions and composable strategy classes for reuse.

意图用于简单的缺失环节过程，而不是重新发明更适合的、最先进的、快速的库，
例如 TA-Lib、Tulipy、PyAlgoTrade、NumPy、SciPy 等。

有关添加建议，请在 [issue tracker] 中提出。

[issue tracker]: https://github.com/kernc/backtesting.py
"""

from __future__ import annotations

import warnings
from collections import OrderedDict
from inspect import currentframe
from itertools import chain, compress, count
from numbers import Number
from typing import Callable, Generator, Optional, Sequence, Union

import numpy as np
import pandas as pd

# 导入绘图和统计模块中的函数
from ._plotting import plot_heatmaps as _plot_heatmaps
from ._stats import compute_stats as _compute_stats
from ._util import SharedMemoryManager, _Array, _as_str, _batch, _tqdm, patch
from .backtesting import Backtest, Strategy

__pdoc__ = {}

# 定义 OHLCV 数据聚合规则（如 Open 取第一个值，High 取最大值等）
OHLCV_AGG = OrderedDict((
    ('Open', 'first'),
    ('High', 'max'),
    ('Low', 'min'),
    ('Close', 'last'),
    ('Volume', 'sum'),
))
"""
字典形式的规则，用于聚合重采样后的 OHLCV 数据帧，例如：
    
    df.resample('4H', label='right').agg(OHLCV_AGG).dropna()
"""

# 定义交易数据聚合规则
TRADES_AGG = OrderedDict((
    ('Size', 'sum'),
    ('EntryBar', 'first'),
    ('ExitBar', 'last'),
    ('EntryPrice', 'mean'),
    ('ExitPrice', 'mean'),
    ('PnL', 'sum'),
    ('ReturnPct', 'mean'),
    ('EntryTime', 'first'),
    ('ExitTime', 'last'),
    ('Duration', 'sum'),
))
"""
字典形式的规则，用于聚合重采样后的交易数据，例如：

    stats['_trades'].resample('1D', on='ExitTime',
                              label='right').agg(TRADES_AGG)
"""

# 权益曲线聚合规则
_EQUITY_AGG = {
    'Equity': 'last',
    'DrawdownPct': 'max',
    'DrawdownDuration': 'max',
}


def barssince(condition: Sequence[bool], default=np.inf) -> int:
    """
    返回 [condition](file://D:\dev\dev_123\backtesting.py\venv\Lib\site-packages\pandas\io\pytables.py#L0-L0) 序列上次为 True 的 K 线数，如果没有则返回 [default](file://D:\dev\dev_123\backtesting.py\venv\Lib\site-packages\bokeh\util\token.py#L231-L234)。

        >>> barssince(self.data.Close > self.data.Open)
        3
    """
    return next(compress(range(len(condition)), reversed(condition)), default)


def cross(series1: Sequence, series2: Sequence) -> bool:
    """
    如果 `series1` 和 `series2` 刚刚交叉（上穿或下穿），返回 True。

        >>> cross(self.data.Close, self.sma)
        True
    """
    return crossover(series1, series2) or crossover(series2, series1)


def crossover(series1: Sequence, series2: Sequence) -> bool:
    """
    如果 `series1` 刚刚上穿 `series2`，返回 True。

        >>> crossover(self.data.Close, self.sma)
        True
    """
    # 处理不同输入类型（如 Series 或数字）
    series1 = (
        series1.values if isinstance(series1, pd.Series) else
        (series1, series1) if isinstance(series1, Number) else
        series1)
    series2 = (
        series2.values if isinstance(series2, pd.Series) else
        (series2, series2) if isinstance(series2, Number) else
        series2)
    try:
        return series1[-2] < series2[-2] and series1[-1] > series2[-1]  # type: ignore
    except IndexError:
        return False


def plot_heatmaps(heatmap: pd.Series,
                  agg: Union[str, Callable] = 'max',
                  *,
                  ncols: int = 3,
                  plot_width: int = 1200,
                  filename: str = '',
                  open_browser: bool = True):
    """
    绘制热力图网格，每个参数对一个热力图。参见 [教程示例]。

    [教程示例]: https://kernc.github.io/backtesting.py/doc/examples/Parameter%20Heatmap%20&%20Optimization.html#plot-heatmap

    `heatmap` 是 [Backtest.optimize(..., return_heatmap=True)](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L1352-L1610) 返回的 Series。

    当将 n 维（n > 2）热力图投影到二维时，默认使用 `'max'` 函数进行聚合。可以通过 [agg](file://D:\dev\dev_123\backtesting.py\venv\Lib\site-packages\pandas\core\groupby\generic.py#L315-L315) 参数自定义。
    """
    return _plot_heatmaps(heatmap, agg, ncols, filename, plot_width, open_browser)


def quantile(series: Sequence, quantile: Union[None, float] = None):
    """
    如果 [quantile](file://D:\dev\dev_123\backtesting.py\backtesting\lib.py#L149-L170) 为 [None](file://D:\dev\dev_123\backtesting.py\venv\Lib\site-packages\bokeh\server\static\js\lib\core\dom.d.ts#L326-L326)，返回 `series` 最后一个值相对于前面值的分位数排名。

    如果 [quantile](file://D:\dev\dev_123\backtesting.py\backtesting\lib.py#L149-L170) 是介于 0 和 1 之间的值，返回该分位数对应的数值。
    百分位可以除以 100 转换为分位数。

        >>> quantile(self.data.Close[-20:], .1)
        162.130
        >>> quantile(self.data.Close)
        0.13
    """
    if quantile is None:
        try:
            last, series = series[-1], series[:-1]
            return np.mean(series < last)
        except IndexError:
            return np.nan
    assert 0 <= quantile <= 1, "quantile 必须在 [0, 1] 范围内"
    return np.nanpercentile(series, quantile * 100)


def compute_stats(
        *,
        stats: pd.Series,
        data: pd.DataFrame,
        trades: pd.DataFrame = None,
        risk_free_rate: float = 0.) -> pd.Series:
    """
    （重新）计算策略性能指标。

    [stats](file://D:\dev\dev_123\backtesting.py\examples\Quick Start User Guide.py#L186-L189) 是 [Backtest.run()](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L1242-L1350) 返回的统计序列。
    [data](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L275-L301) 是传入 [Backtest](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L1083-L1733) 的 OHLC 数据。
    [trades](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L0-L0) 可以是 `stats._trades` 的子集（例如仅多头交易）。
    `risk_free_rate` 用于夏普比率和索提诺比率的计算。

        >>> stats = Backtest(GOOG, MyStrategy).run()
        >>> only_long_trades = stats._trades[stats._trades.Size > 0]
        >>> long_stats = compute_stats(stats=stats, trades=only_long_trades,
        ...                            data=GOOG, risk_free_rate=.02)
    """
    equity = stats._equity_curve.Equity
    if trades is None:
        trades = stats._trades
    else:
        # XXX: 这里可能有 bug？
        equity = equity.copy()
        equity[:] = stats._equity_curve.Equity.iloc[0]
        for t in trades.itertuples(index=False):
            equity.iloc[t.EntryBar:] += t.PnL
    return _compute_stats(trades=trades, equity=equity.values, ohlc_data=data,
                          risk_free_rate=risk_free_rate, strategy_instance=stats._strategy)


def resample_apply(rule: str,
                   func: Optional[Callable[..., Sequence]],
                   series: Union[pd.Series, pd.DataFrame, _Array],
                   *args,
                   agg: Optional[Union[str, dict]] = None,
                   **kwargs):
    """
    将 [func](file://D:\dev\dev_123\backtesting.py\venv\Lib\site-packages\pandas\io\pytables.py#L0-L0)（如指标）应用到按 `rule` 指定的时间周期重采样的 `series` 上。
    如果在 [Strategy.init](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L181-L193) 内部调用，结果会自动包装成 [Strategy.I](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L73-L178) 方法。

    示例：使用 SMA 指标在日线时间框架上操作小时级数据

        class System(Strategy):
            def init(self):
                self.sma = resample_apply('D', SMA, self.data.Close, 10, plot=False)
    """
    if func is None:
        def func(x, *_, **__):
            return x
    assert callable(func), 'resample_apply(func=) 必须是可调用对象'

    if not isinstance(series, (pd.Series, pd.DataFrame)):
        assert isinstance(series, _Array), \
            'resample_apply(series=) 必须是 `pd.Series`, `pd.DataFrame` 或 `Strategy.data.*` 数组'
        series = series.s

    if agg is None:
        agg = OHLCV_AGG.get(getattr(series, 'name', ''), 'last')
        if isinstance(series, pd.DataFrame):
            agg = {column: OHLCV_AGG.get(column, 'last') for column in series.columns}

    resampled = series.resample(rule, label='right').agg(agg).dropna()
    resampled.name = _as_str(series) + '[' + rule + ']'

    # 检查是否在 Strategy.init 中调用
    frame, level = currentframe(), 0
    while frame and level <= 3:
        frame = frame.f_back
        level += 1
        if isinstance(frame.f_locals.get('self'), Strategy):  # type: ignore
            strategy_I = frame.f_locals['self'].I             # type: ignore
            break
    else:
        def strategy_I(func, *args, **kwargs):  # noqa: F811
            return func(*args, **kwargs)

    def wrap_func(resampled, *args, **kwargs):
        result = func(resampled, *args, **kwargs)
        if not isinstance(result, pd.DataFrame) and not isinstance(result, pd.Series):
            result = np.asarray(result)
            if result.ndim == 1:
                result = pd.Series(result, name=resampled.name)
            elif result.ndim == 2:
                result = pd.DataFrame(result.T)
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = resampled.index
        result = result.reindex(index=series.index.union(resampled.index),
                                method='ffill').reindex(series.index)
        return result

    wrap_func.__name__ = func.__name__

    array = strategy_I(wrap_func, resampled, *args, **kwargs)
    return array


def random_ohlc_data(example_data: pd.DataFrame, *,
                     frac=1., random_state: Optional[int] = None) -> Generator[pd.DataFrame, None, None]:
    """
    生成具有类似统计特性的随机 OHLC 数据，可用于压力测试、蒙特卡洛模拟、显著性测试等。

        >>> from backtesting.test import EURUSD
        >>> ohlc_generator = random_ohlc_data(EURUSD)
        >>> next(ohlc_generator)  # 返回新随机数据
        ...
    """
    def shuffle(x):
        return x.sample(frac=frac, replace=frac > 1, random_state=random_state)

    if len(example_data.columns.intersection({'Open', 'High', 'Low', 'Close'})) != 4:
        raise ValueError("[data](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L275-L301) 必须是一个包含 'Open', 'High', 'Low', 'Close' 列的 DataFrame")
    while True:
        df = shuffle(example_data)
        df.index = example_data.index
        padding = df.Close - df.Open.shift(-1)
        gaps = shuffle(example_data.Open.shift(-1) - example_data.Close)
        deltas = (padding + gaps).shift(1).fillna(0).cumsum()
        for key in ('Open', 'High', 'Low', 'Close'):
            df[key] += deltas
        yield df


class SignalStrategy(Strategy):
    """
    一个基于信号的简单辅助策略，支持向量化回测。
    使用方式：继承此类并在 [init()](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L181-L193) 中设置信号。

        class ExampleStrategy(SignalStrategy):
            def init(self):
                super().init()
                self.set_signal(sma1 > sma2, sma1 < sma2)
    """
    __entry_signal = (0,)
    __exit_signal = (False,)

    def set_signal(self, entry_size: Sequence[float],
                   exit_portion: Optional[Sequence[float]] = None,
                   *,
                   plot: bool = True):
        """
        设置入场/离场信号数组。

        `entry_size` > 0 表示多单，< 0 表示空单。
        `exit_portion` 表示要平仓的比例。
        [plot](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L1624-L1733) 表示是否绘制信号。
        """
        self.__entry_signal = self.I(  # type: ignore
            lambda: pd.Series(entry_size, dtype=float).replace(0, np.nan),
            name='entry size', plot=plot, overlay=False, scatter=True, color='black')

        if exit_portion is not None:
            self.__exit_signal = self.I(  # type: ignore
                lambda: pd.Series(exit_portion, dtype=float).replace(0, np.nan),
                name='exit portion', plot=plot, overlay=False, scatter=True, color='black')

    def next(self):
        super().next()

        exit_portion = self.__exit_signal[-1]
        if exit_portion > 0:
            for trade in self.trades:
                if trade.is_long:
                    trade.close(exit_portion)
        elif exit_portion < 0:
            for trade in self.trades:
                if trade.is_short:
                    trade.close(-exit_portion)

        entry_size = self.__entry_signal[-1]
        if entry_size > 0:
            self.buy(size=entry_size)
        elif entry_size < 0:
            self.sell(size=-entry_size)


class TrailingStrategy(Strategy):
    """
    自动跟踪止损策略，根据 ATR（平均真实波幅）设置止损距离。
    使用 `set_trailing_sl()` 设置 ATR 倍数（默认为 6）。
    """
    __n_atr = 6.
    __atr = None

    def init(self):
        super().init()
        self.set_atr_periods()

    def set_atr_periods(self, periods: int = 100):
        """设置 ATR 计算的回溯周期"""
        hi, lo, c_prev = self.data.High, self.data.Low, pd.Series(self.data.Close).shift(1)
        tr = np.max([hi - lo, (c_prev - hi).abs(), (c_prev - lo).abs()], axis=0)
        atr = pd.Series(tr).rolling(periods).mean().bfill().values
        self.__atr = atr

    def set_trailing_sl(self, n_atr: float = 6):
        """设置 ATR 止损倍数"""
        self.__n_atr = n_atr

    def set_trailing_pct(self, pct: float = .05):
        """设置止损百分比（转换为 ATR 单位）"""
        assert 0 < pct < 1, '需要 pct= 作为比例，例如 5% == 0.05'
        pct_in_atr = np.mean(self.data.Close * pct / self.__atr)  # type: ignore
        self.set_trailing_sl(pct_in_atr)

    def next(self):
        super().next()
        index = len(self.data) - 1
        for trade in self.trades:
            if trade.is_long:
                trade.sl = max(trade.sl or -np.inf,
                               self.data.Close[index] - self.__atr[index] * self.__n_atr)
            else:
                trade.sl = min(trade.sl or np.inf,
                               self.data.Close[index] + self.__atr[index] * self.__n_atr)


class FractionalBacktest(Backtest):
    """
    支持小数交易的回测类。
    默认最小单位为 1 satoshi（比特币最小单位），可通过 `fractional_unit` 自定义。
    """
    def __init__(self,
                 data,
                 *args,
                 fractional_unit=1 / 100e6,
                 **kwargs):
        if 'satoshi' in kwargs:
            warnings.warn(
                '参数 `FractionalBacktest(..., satoshi=)` 已弃用。请使用 `FractionalBacktest(..., fractional_unit=)`。',
                category=DeprecationWarning, stacklevel=2)
            fractional_unit = 1 / kwargs.pop('satoshi')
        self._fractional_unit = fractional_unit
        with warnings.catch_warnings(record=True):
            warnings.filterwarnings(action='ignore', message='frac')
            super().__init__(data, *args, **kwargs)

    def run(self, **kwargs) -> pd.Series:
        data = self._data.copy()
        data[['Open', 'High', 'Low', 'Close']] *= self._fractional_unit
        data['Volume'] /= self._fractional_unit
        with patch(self, '_data', data):
            result = super().run(**kwargs)

        trades: pd.DataFrame = result['_trades']
        trades['Size'] *= self._fractional_unit
        trades[['EntryPrice', 'ExitPrice', 'TP', 'SL']] /= self._fractional_unit

        indicators = result['_strategy']._indicators
        for indicator in indicators:
            if indicator._opts['overlay']:
                indicator /= self._fractional_unit

        return result


# 防止 pdoc3 文档化 Strategy 子类的 __init__
for cls in list(globals().values()):
    if isinstance(cls, type) and issubclass(cls, Strategy):
        __pdoc__[f'{cls.__name__}.__init__'] = False


class MultiBacktest:
    """
    多数据集回测封装器，在多个金融工具上并行运行策略。
    示例：

        btm = MultiBacktest([EURUSD, BTCUSD], SmaCross)
        stats_per_ticker: pd.DataFrame = btm.run(fast=10, slow=20)
        heatmap_per_ticker: pd.DataFrame = btm.optimize(...)
    """
    def __init__(self, df_list, strategy_cls, **kwargs):
        self._dfs = df_list
        self._strategy = strategy_cls
        self._bt_kwargs = kwargs

    def run(self, **kwargs):
        """
        并行运行所有数据集，返回带有货币索引的 `pd.DataFrame`
        """
        from . import Pool
        with Pool() as pool, SharedMemoryManager() as smm:
            shm = [smm.df2shm(df) for df in self._dfs]
            results = _tqdm(
                pool.imap(self._mp_task_run,
                          ((df_batch, self._strategy, self._bt_kwargs, kwargs)
                           for df_batch in _batch(shm))),
                total=len(shm),
                desc=self.run.__qualname__,
                mininterval=2
            )
            df = pd.DataFrame(list(chain(*results))).transpose()
        return df

    @staticmethod
    def _mp_task_run(args):
        data_shm, strategy, bt_kwargs, run_kwargs = args
        dfs, shms = zip(*(SharedMemoryManager.shm2df(i) for i in data_shm))
        try:
            return [stats.filter(regex='^[^_]') if stats['# Trades'] else None
                    for stats in (Backtest(df, strategy, **bt_kwargs).run(**run_kwargs)
                                  for df in dfs)]
        finally:
            for shmem in chain(*shms):
                shmem.close()

    def optimize(self, **kwargs) -> pd.DataFrame:
        """
        包装 `Backtest.optimize`，但返回带有货币索引的 `pd.DataFrame`

            heatmap: pd.DataFrame = btm.optimize(...)
            from backtesting.plot import plot_heatmaps
            plot_heatmaps(heatmap.mean(axis=1))
        """
        heatmaps = []
        for df in _tqdm(self._dfs, desc=self.__class__.__name__, mininterval=2):
            bt = Backtest(df, self._strategy, **self._bt_kwargs)
            _best_stats, heatmap = bt.optimize(  # type: ignore
                return_heatmap=True, return_optimization=False, **kwargs)
            heatmaps.append(heatmap)
        heatmap = pd.DataFrame(dict(zip(count(), heatmaps)))
        return heatmap


# 不要在此处之后放置任何代码
__all__ = [getattr(v, '__name__', k)
           for k, v in globals().items()
           if ((callable(v) and v.__module__ == __name__ or
                k.isupper()) and
               not getattr(v, '__name__', k).startswith('_'))]

# 不要在此处之后放置任何代码
