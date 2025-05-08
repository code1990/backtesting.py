"""
核心框架数据结构。
这个模块中的对象也可以直接从顶级模块导入，例如：

    from backtesting import Backtest, Strategy
"""

from __future__ import annotations

import sys
import warnings
from abc import ABCMeta, abstractmethod
from copy import copy
from functools import lru_cache, partial
from itertools import chain, product, repeat
from math import copysign
from numbers import Number
from typing import Callable, List, Optional, Sequence, Tuple, Type, Union

import numpy as np
import pandas as pd
from numpy.random import default_rng

from ._plotting import plot  # noqa: I001
from ._stats import compute_stats, dummy_stats
from ._util import (
    SharedMemoryManager, _as_str, _Indicator, _Data, _batch, _indicator_warmup_nbars,
    _strategy_indicators, patch, try_, _tqdm,
)

__pdoc__ = {
    'Strategy.__init__': False,
    'Order.__init__': False,
    'Position.__init__': False,
    'Trade.__init__': False,
}


class Strategy(metaclass=ABCMeta):
    """
    一个交易策略基类。扩展此类并覆盖方法
    `backtesting.backtesting.Strategy.init` 和
    `backtesting.backtesting.Strategy.next` 来定义你自己的策略。
    """
    def __init__(self, broker, data, params):
        self._indicators = []
        self._broker: _Broker = broker
        self._data: _Data = data
        self._params = self._check_params(params)

    def __repr__(self):
        return '<Strategy ' + str(self) + '>'

    def __str__(self):
        params = ','.join(f'{i[0]}={i[1]}' for i in zip(self._params.keys(),
                                                        map(_as_str, self._params.values())))
        if params:
            params = '(' + params + ')'
        return f'{self.__class__.__name__}{params}'

    def _check_params(self, params):
        for k, v in params.items():
            if not hasattr(self, k):
                raise AttributeError(
                    f"策略 '{self.__class__.__name__}' 缺少参数 '{k}'."
                    "策略类应该在它们可以被优化或运行之前将参数定义为类变量。")
            setattr(self, k, v)
        return params

    def I(self,  # noqa: E743
          func: Callable, *args,
          name=None, plot=True, overlay=None, color=None, scatter=False,
          **kwargs) -> np.ndarray:
        """
        声明一个指标。指标只是一组值（或者在例如MACD指标的情况下是一组数组），
        但在`backtesting.backtesting.Strategy.next`中逐渐揭示，就像
        `backtesting.backtesting.Strategy.data`一样。
        返回`np.ndarray`类型的指标值。

        [func](file://D:\dev\dev_123\backtesting.py\venv\Lib\site-packages\joblib\memory.py#L0-L0) 是一个返回与`backtesting.backtesting.Strategy.data`相同长度的指标数组的函数。

        在图例中，指标用函数名称标记，除非[name](file://D:\dev\dev_123\backtesting.py\venv\Lib\site-packages\six.py#L0-L0)覆盖它。如果[func](file://D:\dev\dev_123\backtesting.py\venv\Lib\site-packages\joblib\memory.py#L0-L0)返回一组数组，
        [name](file://D:\dev\dev_123\backtesting.py\venv\Lib\site-packages\six.py#L0-L0) 可以是一个字符串序列，并且它的大小必须与返回的数组数量一致。

        如果[plot](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L1624-L1733)是`True`，则指标会在`backtesting.backtesting.Backtest.plot`上绘制。

        如果`overlay`是`True`，则指标会叠加在价格蜡烛图上绘制（适合移动平均线等）。
        如果`False`，则指标会在蜡烛图下方单独绘制。默认情况下，使用一个启发式方法来决定。

        [color](file://D:\dev\dev_123\backtesting.py\venv\Lib\site-packages\bokeh\models\css.py#L211-L211)可以是十六进制RGB三元组或X11颜色名称。
        默认情况下，分配下一个可用颜色。

        如果`scatter`是`True`，则绘制的指标标记将是一个圆圈而不是连接的线段（默认）。

        额外的`*args`和`**kwargs`会被传递给[func](file://D:\dev\dev_123\backtesting.py\venv\Lib\site-packages\joblib\memory.py#L0-L0)并可用于参数。

        例如，使用TA-Lib中的简单移动平均函数：

            def init():
                self.sma = self.I(ta.SMA, self.data.Close, self.n_sma)

        .. warning::
            滚动指标可能会以前填充预热值NaN。
            在这种情况下，**回测将仅从所有声明的指标都有非NaN值的第一个条开始**
            （例如，对于使用200条MA的策略，在第201条开始）。
            这可能会影响结果。
        """
        def _format_name(name: str) -> str:
            return name.format(*map(_as_str, args),
                               **dict(zip(kwargs.keys(), map(_as_str, kwargs.values()))))

        if name is None:
            params = ','.join(filter(None, map(_as_str, chain(args, kwargs.values()))))
            func_name = _as_str(func)
            name = (f'{func_name}({params})' if params else f'{func_name}')
        elif isinstance(name, str):
            name = _format_name(name)
        elif try_(lambda: all(isinstance(item, str) for item in name), False):
            name = [_format_name(item) for item in name]
        else:
            raise TypeError(f'意外的 `name=` 类型 {type(name)}; 预期为 [str](file://D:\dev\dev_123\backtesting.py\venv\Lib\site-packages\narwhals\_compliant\expr.py#L870-L871) 或 '
                            '`Sequence[str]`')

        try:
            value = func(*args, **kwargs)
        except Exception as e:
            raise RuntimeError(f'指标 "{name}" 错误。请参见上面的跟踪信息。') from e

        if isinstance(value, pd.DataFrame):
            value = value.values.T

        if value is not None:
            value = try_(lambda: np.asarray(value, order='C'), None)
        is_arraylike = bool(value is not None and value.shape)

        # 可选地翻转数组，如果用户返回了例如`df.values`
        if is_arraylike and np.argmax(value.shape) == 0:
            value = value.T

        if isinstance(name, list) and (np.atleast_2d(value).shape[0] != len(name)):
            raise ValueError(
                f'`name=` 的长度 ({len(name)}) 必须与指标返回的数组数一致 '
                f'({value.shape[0]}).')

        if not is_arraylike or not 1 <= value.ndim <= 2 or value.shape[-1] != len(self._data.Close):
            raise ValueError(
                '指标必须返回(可选的一组)与[data](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L275-L301)长度相同的numpy数组'
                f'(数据形状: {self._data.Close.shape}; 指标 "{name}" '
                f'形状: {getattr(value, "shape", "")}, 返回值: {value})')

        if plot and overlay is None and np.issubdtype(value.dtype, np.number):
            x = value / self._data.Close
            # 默认情况下，如果大多数指标值在Close的30%范围内，则叠加
            with np.errstate(invalid='ignore'):
                overlay = ((x < 1.4) & (x > .6)).mean() > .6

        value = _Indicator(value, name=name, plot=plot, overlay=overlay,
                           color=color, scatter=scatter,
                           # _Indicator.s Series访问器使用此属性：
                           index=self.data.index)
        self._indicators.append(value)
        return value

    @abstractmethod
    def init(self):
        """
        初始化策略。
        覆盖此方法。
        声明指标（使用`backtesting.backtesting.Strategy.I`）。
        预计算需要预先计算或可以在向量化方式下计算的内容。

        如果你扩展了`backtesting.lib`中的可组合策略，
        确保调用:

            super().init()
        """

    @abstractmethod
    def next(self):
        """
        主要的策略运行时方法，每当新的
        `backtesting.backtesting.Strategy.data`
        实例（行；完整蜡烛图条）变得可用时调用。
        这是主要的方法，在其中进行基于在`backtesting.backtesting.Strategy.init`
        中预计算的数据的策略决策。

        如果你扩展了`backtesting.lib`中的可组合策略，
        确保调用:

            super().next()
        """

    class __FULL_EQUITY(float):  # noqa: N801
        def __repr__(self): return '.9999'  # noqa: E704
    _FULL_EQUITY = __FULL_EQUITY(1 - sys.float_info.epsilon)

    def buy(self, *,
            size: float = _FULL_EQUITY,
            limit: Optional[float] = None,
            stop: Optional[float] = None,
            sl: Optional[float] = None,
            tp: Optional[float] = None,
            tag: object = None) -> 'Order':
        """
        放置一个新的多头订单并返回它。有关参数的解释，请参见`Order`及其属性。
        除非你在运行`Backtest(..., trade_on_close=True)`，
        否则市场订单将在下一个条的开盘时成交，
        而其他订单类型（限价单、止损限价单、止损市价单）将在满足相应条件时成交。

        请参阅`Position.close()`和`Trade.close()`以关闭现有仓位。

        参见`Strategy.sell()`。
        """
        assert 0 < size < 1 or round(size) == size >= 1, \
            "size 必须是一个正比例的权益，或一个正整数单位"
        return self._broker.new_order(size, limit, stop, sl, tp, tag)

    def sell(self, *,
             size: float = _FULL_EQUITY,
             limit: Optional[float] = None,
             stop: Optional[float] = None,
             sl: Optional[float] = None,
             tp: Optional[float] = None,
             tag: object = None) -> 'Order':
        """
        放置一个新的空头订单并返回它。有关参数的解释，请参见`Order`及其属性。

        .. caution::
            请注意，`self.sell(size=.1)` 不会关闭现有的 `self.buy(size=.1)`
            交易，除非：

            * 回测是以 `exclusive_orders=True` 运行的，
            * 标的资产价格在两种情况下相等并且
              回测是以 `spread = commission = 0` 运行的。

            使用 `Trade.close()` 或 `Position.close()` 明确退出交易。

        参见 `Strategy.buy()`。

        .. note::
            如果你只是想关闭现有的多头仓位，
            使用 `Position.close()` 或 `Trade.close()`。
        """
        assert 0 < size < 1 or round(size) == size >= 1, \
            "size 必须是一个正比例的权益，或一个正整数单位"
        return self._broker.new_order(-size, limit, stop, sl, tp, tag)

    @property
    def equity(self) -> float:
        """当前账户权益（现金加上资产）。"""
        return self._broker.equity

    @property
    def data(self) -> _Data:
        """
        价格数据，大致如传入到
        `backtesting.backtesting.Backtest.__init__`，
        但有两个显著例外：

        * `data` 不是 DataFrame，而是一个自定义结构
          它提供了定制化的 numpy 数组，为了性能和方便。
          除了 OHLCV 列，`.index` 和长度，
          它还提供 `.pip` 属性，最小的价格变动单位。
        * 在 `backtesting.backtesting.Strategy.init` 内部，`data` 数组
          可以在完整的长度中获得，如传入到
          `backtesting.backtesting.Backtest.__init__`
          （用于预计算指标等）。然而，在
          `backtesting.backtesting.Strategy.next` 内部，`data` 数组
          只有当前迭代那么长，模拟逐步揭示价格点。在每次调用
          `backtesting.backtesting.Strategy.next`（由
          `backtesting.backtesting.Backtest` 内部迭代调用）时，
          最后一个数组值（例如 `data.Close[-1]`）
          总是最新的值。
        * 如果你需要数据数组（例如 `data.Close`）被索引为
          **Pandas series**，你可以调用它们的 `.s` 访问器
          （例如 `data.Close.s`）。如果你需要整个数据作为
          **DataFrame**，请使用 `.df` 访问器（即 `data.df`）。
        """
        return self._data

    @property
    def position(self) -> 'Position':
        """实例化 `backtesting.backtesting.Position`。"""
        return self._broker.position

    @property
    def orders(self) -> 'Tuple[Order, ...]':
        """等待执行的订单列表（参见 `Order`）。"""
        return _Orders(self._broker.orders)

    @property
    def trades(self) -> 'Tuple[Trade, ...]':
        """活动交易的列表（参见 `Trade`）。"""
        return tuple(self._broker.trades)

    @property
    def closed_trades(self) -> 'Tuple[Trade, ...]':
        """已结算交易的列表（参见 `Trade`）。"""
        return tuple(self._broker.closed_trades)



class _Orders(tuple):
    """
    TODO: 删除此类。仅用于弃用。
    """
    def cancel(self):
        """取消所有非或有（即止损/止盈）订单。"""
        for order in self:
            if not order.is_contingent:
                order.cancel()

    def __getattr__(self, item):
        # TODO: 从上一个版本开始警告弃用内容。在下一个版本中删除。
        removed_attrs = ('entry', 'set_entry', 'is_long', 'is_short',
                         'sl', 'tp', 'set_sl', 'set_tp')
        if item in removed_attrs:
            raise AttributeError(f'Strategy.orders.{"/.".join(removed_attrs)} 已在'
                                 'Backtesting 0.2.0 中移除。'
                                 '请使用 [Order](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L380-L534) API。参见文档。')
        raise AttributeError(f"'tuple' 对象没有属性 {item!r}")



class Position:
    """
    当前持有的资产仓位，可在 `backtesting.backtesting.Strategy.next` 中
    通过 `backtesting.backtesting.Strategy.position` 获取。
    可以在布尔上下文中使用，例如：

        if self.position:
            ...  # 我们持有一个仓位，无论是多头还是空头
    """
    def __init__(self, broker: '_Broker'):
        self.__broker = broker

    def __bool__(self):
        return self.size != 0

    @property
    def size(self) -> float:
        """仓位大小（以资产单位表示）。若为负值则表示空头仓位。"""
        return sum(trade.size for trade in self.__broker.trades)

    @property
    def pl(self) -> float:
        """当前仓位的盈亏金额（正值为盈利，负值为亏损）。"""
        return sum(trade.pl for trade in self.__broker.trades)

    @property
    def pl_pct(self) -> float:
        """当前仓位的百分比盈亏（以百分比表示）。"""
        total_invested = sum(trade.entry_price * abs(trade.size) for trade in self.__broker.trades)
        return (self.pl / total_invested) * 100 if total_invested else 0

    @property
    def is_long(self) -> bool:
        """如果仓位是多头（仓位大小为正值），返回 True。"""
        return self.size > 0

    @property
    def is_short(self) -> bool:
        """如果仓位是空头（仓位大小为负值），返回 True。"""
        return self.size < 0

    def close(self, portion: float = 1.):
        """
        关闭部分仓位，通过关闭每个活跃交易的指定比例来实现。
        查看 [Trade.close](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L568-L573) 方法了解更多。
        """
        for trade in self.__broker.trades:
            trade.close(portion)

    def __repr__(self):
        return f'<Position: {self.size} ({len(self.__broker.trades)} trades)>'


class _OutOfMoneyError(Exception):
    """当账户资金不足时抛出的异常"""
    pass


class Order:
    """
    通过 `Strategy.buy()` 和 `Strategy.sell()` 放置新的订单。
    通过 `Strategy.orders` 查询已存在的订单。

    当一个订单被执行或[成交]时，就会产生一个 `Trade`（交易）。

    如果你想修改一个已经下单但尚未成交的订单，
    应该先取消它，然后重新下单。

    所有已下单的订单均为 [Good 'Til Canceled]（长期有效直到成交或手动取消）。

    [filled]: https://www.investopedia.com/terms/f/fill.asp
    [Good 'Til Canceled]: https://www.investopedia.com/terms/g/gtc.asp
    """

    def __init__(self, broker: '_Broker',
                 size: float,
                 limit_price: Optional[float] = None,
                 stop_price: Optional[float] = None,
                 sl_price: Optional[float] = None,
                 tp_price: Optional[float] = None,
                 parent_trade: Optional['Trade'] = None,
                 tag: object = None):
        self.__broker = broker
        assert size != 0
        self.__size = size
        self.__limit_price = limit_price
        self.__stop_price = stop_price
        self.__sl_price = sl_price
        self.__tp_price = tp_price
        self.__parent_trade = parent_trade
        self.__tag = tag

    def _replace(self, **kwargs):
        """
        替换订单的一些属性值。
        """
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self

    def __repr__(self):
        """
        返回订单的字符串表示，用于调试和日志记录。
        """
        return '<Order {}>'.format(', '.join(f'{param}={try_(lambda: round(value, 5), value)!r}'
                                             for param, value in (
                                                 ('size', self.__size),
                                                 ('limit', self.__limit_price),
                                                 ('stop', self.__stop_price),
                                                 ('sl', self.__sl_price),
                                                 ('tp', self.__tp_price),
                                                 ('contingent', self.is_contingent),
                                                 ('tag', self.__tag),
                                             ) if value is not None))

    def cancel(self):
        """
        取消当前订单。
        """
        self.__broker.orders.remove(self)
        trade = self.__parent_trade
        if trade:
            if self is trade._sl_order:
                trade._replace(sl_order=None)
            elif self is trade._tp_order:
                trade._replace(tp_order=None)
            else:
                pass  # 订单由 Trade.close() 下单，无需处理

    # 获取订单字段属性

    @property
    def size(self) -> float:
        """
        订单大小（负值表示空头订单）。

        如果 size 是 0 到 1 之间的值，它将被解释为当前可用流动性的一个比例
        （现金加上仓位盈亏减去已用保证金）。
        值大于等于 1 表示绝对单位数量。
        """
        return self.__size

    @property
    def limit(self) -> Optional[float]:
        """
        [限价单] 的限价，如果是 [市价单] 则为 None。
        限价单将在下一个可用价格成交。

        [限价单]: https://www.investopedia.com/terms/l/limitorder.asp
        [市价单]: https://www.investopedia.com/terms/m/marketorder.asp
        """
        return self.__limit_price

    @property
    def stop(self) -> Optional[float]:
        """
        [止损限价单/止损市价单] 的止损价。
        如果没有设置止损价或者止损价已经被触发，则返回 None。

        [_]: https://www.investopedia.com/terms/s/stoporder.asp
        """
        return self.__stop_price

    @property
    def sl(self) -> Optional[float]:
        """
        若设置了止损价，在执行此订单后，一个新的或有的止损市价单将会在 [Trade](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L539-L711) 上放置。
        参见 [Trade.sl](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L672-L680)。
        """
        return self.__sl_price

    @property
    def tp(self) -> Optional[float]:
        """
        若设置了止盈价，在执行此订单后，一个新的或有的限价单将会在 [Trade](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L539-L711) 上放置。
        参见 [Trade.tp](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L687-L695)。
        """
        return self.__tp_price

    @property
    def parent_trade(self):
        return self.__parent_trade

    @property
    def tag(self):
        """
        随意设置的值（如字符串），如果设置了，可以用来跟踪此订单及其相关的 [Trade](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L539-L711)（参见 [Trade.tag](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L607-L617)）。
        """
        return self.__tag

    __pdoc__['Order.parent_trade'] = False

    # 其他属性

    @property
    def is_long(self):
        """如果订单是多头（订单大小为正值），返回 True。"""
        return self.__size > 0

    @property
    def is_short(self):
        """如果订单是空头（订单大小为负值），返回 True。"""
        return self.__size < 0

    @property
    def is_contingent(self):
        """
        对于[或有订单]返回 True，例如在活跃交易上放置的止损和止盈挂单。
        当其父级 `Trade` 被关闭时，剩余的或有订单也将被取消。

        你可以通过 `Trade.sl` 和 `Trade.tp` 来修改这些或有订单。

        [或有订单]: https://www.investopedia.com/terms/c/contingentorder.asp
        [OCO]: https://www.investopedia.com/terms/o/oco.asp
        """
        return bool((parent := self.__parent_trade) and
                    (self is parent._sl_order or
                     self is parent._tp_order))


class Trade:
    """
    当一个 `Order` 被成交时，就会产生一个活跃的 `Trade`（交易）。

    你可以在 `Strategy.trades` 中找到活跃的交易，
    在 `Strategy.closed_trades` 中找到已关闭和结算的交易。
    """

    def __init__(self, broker: '_Broker', size: int, entry_price: float, entry_bar, tag):
        self.__broker = broker
        self.__size = size
        self.__entry_price = entry_price
        self.__exit_price: Optional[float] = None
        self.__entry_bar: int = entry_bar
        self.__exit_bar: Optional[int] = None
        self.__sl_order: Optional[Order] = None
        self.__tp_order: Optional[Order] = None
        self.__tag = tag
        self._commissions = 0

    def __repr__(self):
        """
        返回交易对象的字符串表示，用于调试和日志记录。
        """
        return f'<Trade size={self.__size} time={self.__entry_bar}-{self.__exit_bar or ""} ' \
               f'price={self.__entry_price}-{self.__exit_price or ""} pl={self.pl:.0f}' \
               f'{" tag=" + str(self.__tag) if self.__tag is not None else ""}>'

    def _replace(self, **kwargs):
        """
        替换交易的一些属性值。
        """
        for k, v in kwargs.items():
            setattr(self, f'_{self.__class__.__qualname__}__{k}', v)
        return self

    def _copy(self, **kwargs):
        """
        创建当前交易的一个副本，并替换部分属性值。
        """
        return copy(self)._replace(**kwargs)

    def close(self, portion: float = 1.):
        """
        放置一个新的订单来关闭当前交易的一部分。

        参数：
            portion (float): 关闭的比例，取值范围为 0 到 1。
        """
        assert 0 < portion <= 1, "portion 必须是介于 0 和 1 之间的分数"
        size = copysign(max(1, round(abs(self.__size) * portion)), -self.__size)
        order = Order(self.__broker, size, parent_trade=self, tag=self.__tag)
        self.__broker.orders.insert(0, order)

    # 获取交易字段属性

    @property
    def size(self):
        """交易大小（正为多头，负为空头）。"""
        return self.__size

    @property
    def entry_price(self) -> float:
        """开仓价格。"""
        return self.__entry_price

    @property
    def exit_price(self) -> Optional[float]:
        """平仓价格（如果交易仍然有效，则返回 None）。"""
        return self.__exit_price

    @property
    def entry_bar(self) -> int:
        """开仓时的价格条索引。"""
        return self.__entry_bar

    @property
    def exit_bar(self) -> Optional[int]:
        """平仓时的价格条索引（如果交易仍然有效，则返回 None）。"""
        return self.__exit_bar

    @property
    def tag(self):
        """
        继承自创建此交易的订单标签。

        可以使用这个标签进行交易跟踪、条件逻辑或子组分析。
        参见 [Order.tag](file://D:\dev\dev_123\backtesting.py\backtesting\backtesting.py#L568-L573)。
        """
        return self.__tag

    @property
    def _sl_order(self):
        """止损订单（内部使用）"""
        return self.__sl_order

    @property
    def _tp_order(self):
        """止盈订单（内部使用）"""
        return self.__tp_order

    # 额外属性

    @property
    def entry_time(self) -> Union[pd.Timestamp, int]:
        """交易开仓时间（基于数据的时间索引）"""
        return self.__broker._data.index[self.__entry_bar]

    @property
    def exit_time(self) -> Optional[Union[pd.Timestamp, int]]:
        """交易平仓时间（基于数据的时间索引）"""
        if self.__exit_bar is None:
            return None
        return self.__broker._data.index[self.__exit_bar]

    @property
    def is_long(self):
        """如果交易是多头（交易大小为正值），返回 True。"""
        return self.__size > 0

    @property
    def is_short(self):
        """如果交易是空头（交易大小为负值），返回 True。"""
        return not self.is_long

    @property
    def pl(self):
        """交易利润（正值）或亏损（负值）金额。"""
        price = self.__exit_price or self.__broker.last_price
        return self.__size * (price - self.__entry_price)

    @property
    def pl_pct(self):
        """交易利润或亏损百分比。"""
        price = self.__exit_price or self.__broker.last_price
        return copysign(1, self.__size) * (price / self.__entry_price - 1)

    @property
    def value(self):
        """交易总价值（体积 × 价格）。"""
        price = self.__exit_price or self.__broker.last_price
        return abs(self.__size) * price

    # 止损/止盈管理 API

    @property
    def sl(self):
        """
        止损价，设置后将生成一个对应的止损订单。

        .. note::
            如果你修改了这个属性，它会取消之前的止损订单并重新下单。
        """
        return self.__sl_order and self.__sl_order.stop

    @sl.setter
    def sl(self, price: float):
        self.__set_contingent('sl', price)

    @property
    def tp(self):
        """
        止盈价，设置后将生成一个对应的止盈订单。

        .. note::
            如果你修改了这个属性，它会取消之前的止盈订单并重新下单。
        """
        return self.__tp_order and self.__tp_order.limit

    @tp.setter
    def tp(self, price: float):
        self.__set_contingent('tp', price)

    def __set_contingent(self, type, price):
        """
        设置或更新一个或有订单（止损/止盈）。

        参数：
            type (str): `'sl'` 表示止损，`'tp'` 表示止盈。
            price (float): 新的价格值。
        """
        assert type in ('sl', 'tp')
        assert price is None or 0 < price < np.inf, f'Make sure 0 < price < inf! price: {price}'
        attr = f'_{self.__class__.__qualname__}__{type}_order'
        order: Order = getattr(self, attr)
        if order:
            order.cancel()
        if price:
            kwargs = {'stop': price} if type == 'sl' else {'limit': price}
            order = self.__broker.new_order(-self.size, trade=self, tag=self.tag, **kwargs)
            setattr(self, attr, order)


class _Broker:
    def __init__(self, *, data, cash, spread, commission, margin,
                 trade_on_close, hedging, exclusive_orders, index):
        assert cash > 0, f"初始现金必须大于0，当前为 {cash}"
        assert 0 < margin <= 1, f"保证金比例应在 (0, 1] 范围内，当前为 {margin}"

        self._data: _Data = data  # 市场数据对象
        self._cash = cash  # 当前现金余额

        # 设置手续费规则：如果是函数则直接使用，否则解析为固定和相对手续费
        if callable(commission):
            self._commission = commission
        else:
            try:
                self._commission_fixed, self._commission_relative = commission
            except TypeError:
                self._commission_fixed, self._commission_relative = 0, commission
            assert self._commission_fixed >= 0, '固定手续费必须 >= 0'
            assert -.1 <= self._commission_relative < .1, \
                ("相对手续费应在 -10% 到 10% 之间，例如市场回扣或平台费率，"
                 f"当前为 {self._commission_relative}")
            self._commission = self._commission_func

        self._spread = spread  # 买卖价差（点差）
        self._leverage = 1 / margin  # 杠杆率 = 1 / 保证金比例
        self._trade_on_close = trade_on_close  # 是否使用收盘价成交
        self._hedging = hedging  # 是否允许对冲（多空同时持仓）
        self._exclusive_orders = exclusive_orders  # 是否开启互斥订单（新订单取消旧订单）

        self._equity = np.tile(np.nan, len(index))  # 账户权益曲线初始化
        self.orders: List[Order] = []  # 当前未成交订单列表
        self.trades: List[Trade] = []  # 当前持仓交易列表
        self.position = Position(self)  # 当前总持仓对象
        self.closed_trades: List[Trade] = []  # 已平仓交易记录列表

    def _commission_func(self, order_size, price):
        # 计算交易手续费：固定费用 + 按交易金额比例计算的费用
        return self._commission_fixed + abs(order_size) * price * self._commission_relative

    def __repr__(self):
        return f'<Broker: {self._cash:.0f}{self.position.pl:+.1f} ({len(self.trades)} trades)>'

    def new_order(self,
                  size: float,
                  limit: Optional[float] = None,
                  stop: Optional[float] = None,
                  sl: Optional[float] = None,
                  tp: Optional[float] = None,
                  tag: object = None,
                  *,
                  trade: Optional[Trade] = None) -> Order:
        """
        创建一个新订单。`size` 表示订单方向和大小，正数为多，负数为空。
        可设置限价、止损、止盈等参数。
        """
        size = float(size)
        stop = stop and float(stop)
        limit = limit and float(limit)
        sl = sl and float(sl)
        tp = tp and float(tp)

        is_long = size > 0
        assert size != 0, size
        adjusted_price = self._adjusted_price(size)

        # 合法性校验：多单价格关系为 SL < Limit/Stop/市价 < TP，空单相反
        if is_long:
            if not (sl or -np.inf) < (limit or stop or adjusted_price) < (tp or np.inf):
                raise ValueError(
                    "多单要求: "
                    f"止损({sl}) < 限价/止损价({limit or stop or adjusted_price}) < 止盈({tp})")
        else:
            if not (tp or -np.inf) < (limit or stop or adjusted_price) < (sl or np.inf):
                raise ValueError(
                    "空单要求: "
                    f"止盈({tp}) < 限价/止损价({limit or stop or adjusted_price}) < 止损({sl})")

        order = Order(self, size, limit, stop, sl, tp, trade, tag)

        if not trade and self._exclusive_orders:
            # 如果开启互斥订单，新订单将取消之前的所有订单并平掉持仓
            for o in self.orders:
                if not o.is_contingent:
                    o.cancel()
            for t in self.trades:
                t.close()

        # 将新订单插入队列；如果是止损订单，优先处理
        self.orders.insert(0 if trade and stop else len(self.orders), order)
        return order

    @property
    def last_price(self) -> float:
        """ 获取最近收盘价 """
        return self._data.Close[-1]

    def _adjusted_price(self, size=None, price=None) -> float:
        """
        返回考虑点差后的价格。多单价格会上浮，空单价格下调。
        """
        return (price or self.last_price) * (1 + copysign(self._spread, size))

    @property
    def equity(self) -> float:
        """ 当前总权益 = 现金 + 所有持仓盈亏 """
        return self._cash + sum(trade.pl for trade in self.trades)

    @property
    def margin_available(self) -> float:
        """ 当前可用保证金 """
        margin_used = sum(trade.value / self._leverage for trade in self.trades)
        return max(0, self.equity - margin_used)

    def next(self):
        """ 执行一次迭代，处理订单，更新账户状态 """
        i = self._i = len(self._data) - 1
        self._process_orders()

        equity = self.equity
        self._equity[i] = equity

        # 账户爆仓处理
        if equity <= 0:
            assert self.margin_available <= 0
            for trade in self.trades:
                self._close_trade(trade, self._data.Close[-1], i)
            self._cash = 0
            self._equity[i:] = 0
            raise _OutOfMoneyError

    def _process_orders(self):
        """
        核心逻辑：逐个处理当前订单队列，判断是否满足触发条件，若满足则成交。
        同时处理止盈/止损等跟随订单。
        """
        data = self._data
        open, high, low = data.Open[-1], data.High[-1], data.Low[-1]
        reprocess_orders = False

        for order in list(self.orders):

            if order not in self.orders:
                continue

            stop_price = order.stop
            if stop_price:
                is_stop_hit = ((high >= stop_price) if order.is_long else (low <= stop_price))
                if not is_stop_hit:
                    continue
                order._replace(stop_price=None)

            # 判断限价订单是否触发
            if order.limit:
                is_limit_hit = low <= order.limit if order.is_long else high >= order.limit
                is_limit_hit_before_stop = (is_limit_hit and
                                            (order.limit <= (stop_price or -np.inf)
                                             if order.is_long
                                             else order.limit >= (stop_price or np.inf)))
                if not is_limit_hit or is_limit_hit_before_stop:
                    continue

                price = (min(stop_price or open, order.limit)
                         if order.is_long else
                         max(stop_price or open, order.limit))
            else:
                prev_close = data.Close[-2]
                price = prev_close if self._trade_on_close and not order.is_contingent else open
                if stop_price:
                    price = max(price, stop_price) if order.is_long else min(price, stop_price)

            is_market_order = not order.limit and not stop_price
            time_index = (
                (self._i - 1)
                if is_market_order and self._trade_on_close and not order.is_contingent else
                self._i)

            # 如果是跟随订单（止盈/止损），关闭原始交易
            if order.parent_trade:
                trade = order.parent_trade
                _prev_size = trade.size
                size = copysign(min(abs(_prev_size), abs(order.size)), order.size)
                if trade in self.trades:
                    self._reduce_trade(trade, price, size, time_index)
                    assert order.size != -_prev_size or trade not in self.trades
                if order in (trade._sl_order, trade._tp_order):
                    assert order.size == -trade.size
                    assert order not in self.orders
                else:
                    assert abs(_prev_size) >= abs(size) >= 1
                    self.orders.remove(order)
                continue

            # 独立订单：判断是否满足下单条件
            adjusted_price = self._adjusted_price(order.size, price)
            adjusted_price_plus_commission = adjusted_price + self._commission(order.size, price)

            size = order.size
            if -1 < size < 1:
                size = copysign(int((self.margin_available * self._leverage * abs(size))
                                    // adjusted_price_plus_commission), size)
                if not size:
                    warnings.warn(
                        f'time={self._i}: 相对规模订单由于可用保证金不足被取消。', category=UserWarning)
                    self.orders.remove(order)
                    continue
            assert size == round(size)
            need_size = int(size)

            if not self._hedging:
                for trade in list(self.trades):
                    if trade.is_long == order.is_long:
                        continue
                    if abs(need_size) >= abs(trade.size):
                        self._close_trade(trade, price, time_index)
                        need_size += trade.size
                    else:
                        self._reduce_trade(trade, price, need_size, time_index)
                        need_size = 0
                    if not need_size:
                        break

            if abs(need_size) * adjusted_price_plus_commission > self.margin_available * self._leverage:
                self.orders.remove(order)
                continue

            if need_size:
                self._open_trade(adjusted_price,
                                 need_size,
                                 order.sl,
                                 order.tp,
                                 time_index,
                                 order.tag)

                if order.sl or order.tp:
                    if is_market_order:
                        reprocess_orders = True
                    elif stop_price and not order.limit and order.tp and (
                            (order.is_long and order.tp <= high and (order.sl or -np.inf) < low) or
                            (order.is_short and order.tp >= low and (order.sl or np.inf) > high)):
                        reprocess_orders = True
                    elif (low <= (order.sl or -np.inf) <= high or
                          low <= (order.tp or -np.inf) <= high):
                        warnings.warn(
                            f"({data.index[-1]}) 同一根K线内的触发订单及其止盈/止损存在冲突，"
                            "止盈/止损将在下一根K线处理。此结果可能不具确定性。", UserWarning)

            self.orders.remove(order)

        if reprocess_orders:
            self._process_orders()

    def _reduce_trade(self, trade: Trade, price: float, size: float, time_index: int):
        """ 减少或部分平掉某笔持仓 """
        assert trade.size * size < 0
        assert abs(trade.size) >= abs(size)

        size_left = trade.size + size
        assert size_left * trade.size >= 0
        if not size_left:
            close_trade = trade
        else:
            trade._replace(size=size_left)
            if trade._sl_order:
                trade._sl_order._replace(size=-trade.size)
            if trade._tp_order:
                trade._tp_order._replace(size=-trade.size)

            close_trade = trade._copy(size=-size, sl_order=None, tp_order=None)
            self.trades.append(close_trade)

        self._close_trade(close_trade, price, time_index)

    def _close_trade(self, trade: Trade, price: float, time_index: int):
        """ 完全平掉一个交易 """
        self.trades.remove(trade)
        if trade._sl_order:
            self.orders.remove(trade._sl_order)
        if trade._tp_order:
            self.orders.remove(trade._tp_order)

        closed_trade = trade._replace(exit_price=price, exit_bar=time_index)
        self.closed_trades.append(closed_trade)

        # 平仓时扣除手续费
        commission = self._commission(trade.size, price)
        self._cash += trade.pl - commission

        trade_open_commission = self._commission(closed_trade.size, closed_trade.entry_price)
        closed_trade._commissions = commission + trade_open_commission

    def _open_trade(self, price: float, size: int,
                    sl: Optional[float], tp: Optional[float], time_index: int, tag):
        """ 开仓逻辑，生成新的交易对象 """
        trade = Trade(self, size, price, time_index, tag)
        self.trades.append(trade)
        self._cash -= self._commission(size, price)
        if tp:
            trade.tp = tp
        if sl:
            trade.sl = sl


class Backtest:
    """
    Backtest a particular (parameterized) strategy
    on particular data.

    Initialize a backtest. Requires data and a strategy to test.
    After initialization, you can call method
    `backtesting.backtesting.Backtest.run` to run a backtest
    instance, or `backtesting.backtesting.Backtest.optimize` to
    optimize it.

    `data` is a `pd.DataFrame` with columns:
    `Open`, `High`, `Low`, `Close`, and (optionally) `Volume`.
    If any columns are missing, set them to what you have available,
    e.g.

        df['Open'] = df['High'] = df['Low'] = df['Close']

    The passed data frame can contain additional columns that
    can be used by the strategy (e.g. sentiment info).
    DataFrame index can be either a datetime index (timestamps)
    or a monotonic range index (i.e. a sequence of periods).

    `strategy` is a `backtesting.backtesting.Strategy`
    _subclass_ (not an instance).

    `cash` is the initial cash to start with.

    `spread` is the the constant bid-ask spread rate (relative to the price).
    E.g. set it to `0.0002` for commission-less forex
    trading where the average spread is roughly 0.2‰ of the asking price.

    `commission` is the commission rate. E.g. if your broker's commission
    is 1% of order value, set commission to `0.01`.
    The commission is applied twice: at trade entry and at trade exit.
    Besides one single floating value, `commission` can also be a tuple of floating
    values `(fixed, relative)`. E.g. set it to `(100, .01)`
    if your broker charges minimum $100 + 1%.
    Additionally, `commission` can be a callable
    `func(order_size: int, price: float) -> float`
    (note, order size is negative for short orders),
    which can be used to model more complex commission structures.
    Negative commission values are interpreted as market-maker's rebates.

    .. note::
        Before v0.4.0, the commission was only applied once, like `spread` is now.
        If you want to keep the old behavior, simply set `spread` instead.

    .. note::
        With nonzero `commission`, long and short orders will be placed
        at an adjusted price that is slightly higher or lower (respectively)
        than the current price. See e.g.
        [#153](https://github.com/kernc/backtesting.py/issues/153),
        [#538](https://github.com/kernc/backtesting.py/issues/538),
        [#633](https://github.com/kernc/backtesting.py/issues/633).

    `margin` is the required margin (ratio) of a leveraged account.
    No difference is made between initial and maintenance margins.
    To run the backtest using e.g. 50:1 leverge that your broker allows,
    set margin to `0.02` (1 / leverage).

    If `trade_on_close` is `True`, market orders will be filled
    with respect to the current bar's closing price instead of the
    next bar's open.

    If `hedging` is `True`, allow trades in both directions simultaneously.
    If `False`, the opposite-facing orders first close existing trades in
    a [FIFO] manner.

    If `exclusive_orders` is `True`, each new order auto-closes the previous
    trade/position, making at most a single trade (long or short) in effect
    at each time.

    If `finalize_trades` is `True`, the trades that are still
    [active and ongoing] at the end of the backtest will be closed on
    the last bar and will contribute to the computed backtest statistics.

    .. tip:: Fractional trading
        See also `backtesting.lib.FractionalBacktest` if you want to trade
        fractional units (of e.g. bitcoin).

    [FIFO]: https://www.investopedia.com/terms/n/nfa-compliance-rule-2-43b.asp
    [active and ongoing]: https://kernc.github.io/backtesting.py/doc/backtesting/backtesting.html#backtesting.backtesting.Strategy.trades
    """  # noqa: E501
    def __init__(self,
                 data: pd.DataFrame,
                 strategy: Type[Strategy],
                 *,
                 cash: float = 10_000,
                 spread: float = .0,
                 commission: Union[float, Tuple[float, float]] = .0,
                 margin: float = 1.,
                 trade_on_close=False,
                 hedging=False,
                 exclusive_orders=False,
                 finalize_trades=False,
                 ):
        if not (isinstance(strategy, type) and issubclass(strategy, Strategy)):
            raise TypeError('`strategy` must be a Strategy sub-type')
        if not isinstance(data, pd.DataFrame):
            raise TypeError("`data` must be a pandas.DataFrame with columns")
        if not isinstance(spread, Number):
            raise TypeError('`spread` must be a float value, percent of '
                            'entry order price')
        if not isinstance(commission, (Number, tuple)) and not callable(commission):
            raise TypeError('`commission` must be a float percent of order value, '
                            'a tuple of `(fixed, relative)` commission, '
                            'or a function that takes `(order_size, price)`'
                            'and returns commission dollar value')

        data = data.copy(deep=False)

        # Convert index to datetime index
        if (not isinstance(data.index, pd.DatetimeIndex) and
            not isinstance(data.index, pd.RangeIndex) and
            # Numeric index with most large numbers
            (data.index.is_numeric() and
             (data.index > pd.Timestamp('1975').timestamp()).mean() > .8)):
            try:
                data.index = pd.to_datetime(data.index, infer_datetime_format=True)
            except ValueError:
                pass

        if 'Volume' not in data:
            data['Volume'] = np.nan

        if len(data) == 0:
            raise ValueError('OHLC `data` is empty')
        if len(data.columns.intersection({'Open', 'High', 'Low', 'Close', 'Volume'})) != 5:
            raise ValueError("`data` must be a pandas.DataFrame with columns "
                             "'Open', 'High', 'Low', 'Close', and (optionally) 'Volume'")
        if data[['Open', 'High', 'Low', 'Close']].isnull().values.any():
            raise ValueError('Some OHLC values are missing (NaN). '
                             'Please strip those lines with `df.dropna()` or '
                             'fill them in with `df.interpolate()` or whatever.')
        if np.any(data['Close'] > cash):
            warnings.warn('Some prices are larger than initial cash value. Note that fractional '
                          'trading is not supported. If you want to trade Bitcoin, '
                          'increase initial cash, or trade μBTC or satoshis instead (GH-134).',
                          stacklevel=2)
        if not data.index.is_monotonic_increasing:
            warnings.warn('Data index is not sorted in ascending order. Sorting.',
                          stacklevel=2)
            data = data.sort_index()
        if not isinstance(data.index, pd.DatetimeIndex):
            warnings.warn('Data index is not datetime. Assuming simple periods, '
                          'but `pd.DateTimeIndex` is advised.',
                          stacklevel=2)

        self._data: pd.DataFrame = data
        self._broker = partial(
            _Broker, cash=cash, spread=spread, commission=commission, margin=margin,
            trade_on_close=trade_on_close, hedging=hedging,
            exclusive_orders=exclusive_orders, index=data.index,
        )
        self._strategy = strategy
        self._results: Optional[pd.Series] = None
        self._finalize_trades = bool(finalize_trades)

    def run(self, **kwargs) -> pd.Series:
        """
        Run the backtest. Returns `pd.Series` with results and statistics.

        Keyword arguments are interpreted as strategy parameters.

            >>> Backtest(GOOG, SmaCross).run()
            Start                     2004-08-19 00:00:00
            End                       2013-03-01 00:00:00
            Duration                   3116 days 00:00:00
            Exposure Time [%]                    96.74115
            Equity Final [$]                     51422.99
            Equity Peak [$]                      75787.44
            Return [%]                           414.2299
            Buy & Hold Return [%]               703.45824
            Return (Ann.) [%]                    21.18026
            Volatility (Ann.) [%]                36.49391
            CAGR [%]                             14.15984
            Sharpe Ratio                          0.58038
            Sortino Ratio                         1.08479
            Calmar Ratio                          0.44144
            Alpha [%]                           394.37391
            Beta                                  0.03803
            Max. Drawdown [%]                   -47.98013
            Avg. Drawdown [%]                    -5.92585
            Max. Drawdown Duration      584 days 00:00:00
            Avg. Drawdown Duration       41 days 00:00:00
            # Trades                                   66
            Win Rate [%]                          46.9697
            Best Trade [%]                       53.59595
            Worst Trade [%]                     -18.39887
            Avg. Trade [%]                        2.53172
            Max. Trade Duration         183 days 00:00:00
            Avg. Trade Duration          46 days 00:00:00
            Profit Factor                         2.16795
            Expectancy [%]                        3.27481
            SQN                                   1.07662
            Kelly Criterion                       0.15187
            _strategy                            SmaCross
            _equity_curve                           Eq...
            _trades                       Size  EntryB...
            dtype: object

        .. warning::
            You may obtain different results for different strategy parameters.
            E.g. if you use 50- and 200-bar SMA, the trading simulation will
            begin on bar 201. The actual length of delay is equal to the lookback
            period of the `Strategy.I` indicator which lags the most.
            Obviously, this can affect results.
        """
        data = _Data(self._data.copy(deep=False))
        broker: _Broker = self._broker(data=data)
        strategy: Strategy = self._strategy(broker, data, kwargs)

        strategy.init()
        data._update()  # Strategy.init might have changed/added to data.df

        # Indicators used in Strategy.next()
        indicator_attrs = _strategy_indicators(strategy)

        # Skip first few candles where indicators are still "warming up"
        # +1 to have at least two entries available
        start = 1 + _indicator_warmup_nbars(strategy)

        # Disable "invalid value encountered in ..." warnings. Comparison
        # np.nan >= 3 is not invalid; it's False.
        with np.errstate(invalid='ignore'):

            for i in _tqdm(range(start, len(self._data)), desc=self.run.__qualname__,
                           unit='bar', mininterval=2, miniters=100):
                # Prepare data and indicators for `next` call
                data._set_length(i + 1)
                for attr, indicator in indicator_attrs:
                    # Slice indicator on the last dimension (case of 2d indicator)
                    setattr(strategy, attr, indicator[..., :i + 1])

                # Handle orders processing and broker stuff
                try:
                    broker.next()
                except _OutOfMoneyError:
                    break

                # Next tick, a moment before bar close
                strategy.next()
            else:
                if self._finalize_trades is True:
                    # Close any remaining open trades so they produce some stats
                    for trade in reversed(broker.trades):
                        trade.close()

                    # HACK: Re-run broker one last time to handle close orders placed in the last
                    #  strategy iteration. Use the same OHLC values as in the last broker iteration.
                    if start < len(self._data):
                        try_(broker.next, exception=_OutOfMoneyError)

            # Set data back to full length
            # for future `indicator._opts['data'].index` calls to work
            data._set_length(len(self._data))

            equity = pd.Series(broker._equity).bfill().fillna(broker._cash).values
            self._results = compute_stats(
                trades=broker.closed_trades,
                equity=equity,
                ohlc_data=self._data,
                risk_free_rate=0.0,
                strategy_instance=strategy,
            )

        return self._results

    def optimize(self, *,
                 maximize: Union[str, Callable[[pd.Series], float]] = 'SQN',
                 method: str = 'grid',
                 max_tries: Optional[Union[int, float]] = None,
                 constraint: Optional[Callable[[dict], bool]] = None,
                 return_heatmap: bool = False,
                 return_optimization: bool = False,
                 random_state: Optional[int] = None,
                 **kwargs) -> Union[pd.Series,
                                    Tuple[pd.Series, pd.Series],
                                    Tuple[pd.Series, pd.Series, dict]]:
        """
        Optimize strategy parameters to an optimal combination.
        Returns result `pd.Series` of the best run.

        `maximize` is a string key from the
        `backtesting.backtesting.Backtest.run`-returned results series,
        or a function that accepts this series object and returns a number;
        the higher the better. By default, the method maximizes
        Van Tharp's [System Quality Number](https://google.com/search?q=System+Quality+Number).

        `method` is the optimization method. Currently two methods are supported:

        * `"grid"` which does an exhaustive (or randomized) search over the
          cartesian product of parameter combinations, and
        * `"sambo"` which finds close-to-optimal strategy parameters using
          [model-based optimization], making at most `max_tries` evaluations.

        [model-based optimization]: https://sambo-optimization.github.io

        `max_tries` is the maximal number of strategy runs to perform.
        If `method="grid"`, this results in randomized grid search.
        If `max_tries` is a floating value between (0, 1], this sets the
        number of runs to approximately that fraction of full grid space.
        Alternatively, if integer, it denotes the absolute maximum number
        of evaluations. If unspecified (default), grid search is exhaustive,
        whereas for `method="sambo"`, `max_tries` is set to 200.

        `constraint` is a function that accepts a dict-like object of
        parameters (with values) and returns `True` when the combination
        is admissible to test with. By default, any parameters combination
        is considered admissible.

        If `return_heatmap` is `True`, besides returning the result
        series, an additional `pd.Series` is returned with a multiindex
        of all admissible parameter combinations, which can be further
        inspected or projected onto 2D to plot a heatmap
        (see `backtesting.lib.plot_heatmaps()`).

        If `return_optimization` is True and `method = 'sambo'`,
        in addition to result series (and maybe heatmap), return raw
        [`scipy.optimize.OptimizeResult`][OptimizeResult] for further
        inspection, e.g. with [SAMBO]'s [plotting tools].

        [OptimizeResult]: https://sambo-optimization.github.io/doc/sambo/#sambo.OptimizeResult
        [SAMBO]: https://sambo-optimization.github.io
        [plotting tools]: https://sambo-optimization.github.io/doc/sambo/plot.html

        If you want reproducible optimization results, set `random_state`
        to a fixed integer random seed.

        Additional keyword arguments represent strategy arguments with
        list-like collections of possible values. For example, the following
        code finds and returns the "best" of the 7 admissible (of the
        9 possible) parameter combinations:

            best_stats = backtest.optimize(sma1=[5, 10, 15], sma2=[10, 20, 40],
                                           constraint=lambda p: p.sma1 < p.sma2)
        """
        if not kwargs:
            raise ValueError('Need some strategy parameters to optimize')

        maximize_key = None
        if isinstance(maximize, str):
            maximize_key = str(maximize)
            if maximize not in dummy_stats().index:
                raise ValueError('`maximize`, if str, must match a key in pd.Series '
                                 'result of backtest.run()')

            def maximize(stats: pd.Series, _key=maximize):
                return stats[_key]

        elif not callable(maximize):
            raise TypeError('`maximize` must be str (a field of backtest.run() result '
                            'Series) or a function that accepts result Series '
                            'and returns a number; the higher the better')
        assert callable(maximize), maximize

        have_constraint = bool(constraint)
        if constraint is None:

            def constraint(_):
                return True

        elif not callable(constraint):
            raise TypeError("`constraint` must be a function that accepts a dict "
                            "of strategy parameters and returns a bool whether "
                            "the combination of parameters is admissible or not")
        assert callable(constraint), constraint

        if method == 'skopt':
            method = 'sambo'
            warnings.warn('`Backtest.optimize(method="skopt")` is deprecated. Use `method="sambo"`.',
                          DeprecationWarning, stacklevel=2)
        if return_optimization and method != 'sambo':
            raise ValueError("return_optimization=True only valid if method='sambo'")

        def _tuple(x):
            return x if isinstance(x, Sequence) and not isinstance(x, str) else (x,)

        for k, v in kwargs.items():
            if len(_tuple(v)) == 0:
                raise ValueError(f"Optimization variable '{k}' is passed no "
                                 f"optimization values: {k}={v}")

        class AttrDict(dict):
            def __getattr__(self, item):
                return self[item]

        def _grid_size():
            size = int(np.prod([len(_tuple(v)) for v in kwargs.values()]))
            if size < 10_000 and have_constraint:
                size = sum(1 for p in product(*(zip(repeat(k), _tuple(v))
                                                for k, v in kwargs.items()))
                           if constraint(AttrDict(p)))
            return size

        def _optimize_grid() -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
            rand = default_rng(random_state).random
            grid_frac = (1 if max_tries is None else
                         max_tries if 0 < max_tries <= 1 else
                         max_tries / _grid_size())
            param_combos = [dict(params)  # back to dict so it pickles
                            for params in (AttrDict(params)
                                           for params in product(*(zip(repeat(k), _tuple(v))
                                                                   for k, v in kwargs.items())))
                            if constraint(params)
                            and rand() <= grid_frac]
            if not param_combos:
                raise ValueError('No admissible parameter combinations to test')

            if len(param_combos) > 300:
                warnings.warn(f'Searching for best of {len(param_combos)} configurations.',
                              stacklevel=2)

            heatmap = pd.Series(np.nan,
                                name=maximize_key,
                                index=pd.MultiIndex.from_tuples(
                                    [p.values() for p in param_combos],
                                    names=next(iter(param_combos)).keys()))

            from . import Pool
            with Pool() as pool, \
                    SharedMemoryManager() as smm:
                with patch(self, '_data', None):
                    bt = copy(self)  # bt._data will be reassigned in _mp_task worker
                results = _tqdm(
                    pool.imap(Backtest._mp_task,
                              ((bt, smm.df2shm(self._data), params_batch)
                               for params_batch in _batch(param_combos))),
                    total=len(param_combos),
                    desc='Backtest.optimize'
                )
                for param_batch, result in zip(_batch(param_combos), results):
                    for params, stats in zip(param_batch, result):
                        if stats is not None:
                            heatmap[tuple(params.values())] = maximize(stats)

            if pd.isnull(heatmap).all():
                # No trade was made in any of the runs. Just make a random
                # run so we get some, if empty, results
                stats = self.run(**param_combos[0])
            else:
                best_params = heatmap.idxmax(skipna=True)
                stats = self.run(**dict(zip(heatmap.index.names, best_params)))

            if return_heatmap:
                return stats, heatmap
            return stats

        def _optimize_sambo() -> Union[pd.Series,
                                       Tuple[pd.Series, pd.Series],
                                       Tuple[pd.Series, pd.Series, dict]]:
            try:
                import sambo
            except ImportError:
                raise ImportError("Need package 'sambo' for method='sambo'. pip install sambo") from None

            nonlocal max_tries
            max_tries = (200 if max_tries is None else
                         max(1, int(max_tries * _grid_size())) if 0 < max_tries <= 1 else
                         max_tries)

            dimensions = []
            for key, values in kwargs.items():
                values = np.asarray(values)
                if values.dtype.kind in 'mM':  # timedelta, datetime64
                    # these dtypes are unsupported in SAMBO, so convert to raw int
                    # TODO: save dtype and convert back later
                    values = values.astype(np.int64)

                if values.dtype.kind in 'iumM':
                    dimensions.append((values.min(), values.max() + 1))
                elif values.dtype.kind == 'f':
                    dimensions.append((values.min(), values.max()))
                else:
                    dimensions.append(values.tolist())

            # Avoid recomputing re-evaluations
            @lru_cache()
            def memoized_run(tup):
                nonlocal maximize, self
                stats = self.run(**dict(tup))
                return -maximize(stats)

            progress = iter(_tqdm(repeat(None), total=max_tries, leave=False,
                                  desc=self.optimize.__qualname__, mininterval=2))
            _names = tuple(kwargs.keys())

            def objective_function(x):
                nonlocal progress, memoized_run, constraint, _names
                next(progress)
                value = memoized_run(tuple(zip(_names, x)))
                return 0 if np.isnan(value) else value

            def cons(x):
                nonlocal constraint, _names
                return constraint(AttrDict(zip(_names, x)))

            res = sambo.minimize(
                fun=objective_function,
                bounds=dimensions,
                constraints=cons,
                max_iter=max_tries,
                method='sceua',
                rng=random_state)

            stats = self.run(**dict(zip(kwargs.keys(), res.x)))
            output = [stats]

            if return_heatmap:
                heatmap = pd.Series(dict(zip(map(tuple, res.xv), -res.funv)),
                                    name=maximize_key)
                heatmap.index.names = kwargs.keys()
                heatmap.sort_index(inplace=True)
                output.append(heatmap)

            if return_optimization:
                output.append(res)

            return stats if len(output) == 1 else tuple(output)

        if method == 'grid':
            output = _optimize_grid()
        elif method in ('sambo', 'skopt'):
            output = _optimize_sambo()
        else:
            raise ValueError(f"Method should be 'grid' or 'sambo', not {method!r}")
        return output

    @staticmethod
    def _mp_task(arg):
        bt, data_shm, params_batch = arg
        bt._data, shm = SharedMemoryManager.shm2df(data_shm)
        try:
            return [stats.filter(regex='^[^_]') if stats['# Trades'] else None
                    for stats in (bt.run(**params)
                                  for params in params_batch)]
        finally:
            for shmem in shm:
                shmem.close()

    def plot(self, *, results: pd.Series = None, filename=None, plot_width=None,
             plot_equity=True, plot_return=False, plot_pl=True,
             plot_volume=True, plot_drawdown=False, plot_trades=True,
             smooth_equity=False, relative_equity=True,
             superimpose: Union[bool, str] = True,
             resample=True, reverse_indicators=False,
             show_legend=True, open_browser=True):
        """
        Plot the progression of the last backtest run.

        If `results` is provided, it should be a particular result
        `pd.Series` such as returned by
        `backtesting.backtesting.Backtest.run` or
        `backtesting.backtesting.Backtest.optimize`, otherwise the last
        run's results are used.

        `filename` is the path to save the interactive HTML plot to.
        By default, a strategy/parameter-dependent file is created in the
        current working directory.

        `plot_width` is the width of the plot in pixels. If None (default),
        the plot is made to span 100% of browser width. The height is
        currently non-adjustable.

        If `plot_equity` is `True`, the resulting plot will contain
        an equity (initial cash plus assets) graph section. This is the same
        as `plot_return` plus initial 100%.

        If `plot_return` is `True`, the resulting plot will contain
        a cumulative return graph section. This is the same
        as `plot_equity` minus initial 100%.

        If `plot_pl` is `True`, the resulting plot will contain
        a profit/loss (P/L) indicator section.

        If `plot_volume` is `True`, the resulting plot will contain
        a trade volume section.

        If `plot_drawdown` is `True`, the resulting plot will contain
        a separate drawdown graph section.

        If `plot_trades` is `True`, the stretches between trade entries
        and trade exits are marked by hash-marked tractor beams.

        If `smooth_equity` is `True`, the equity graph will be
        interpolated between fixed points at trade closing times,
        unaffected by any interim asset volatility.

        If `relative_equity` is `True`, scale and label equity graph axis
        with return percent, not absolute cash-equivalent values.

        If `superimpose` is `True`, superimpose larger-timeframe candlesticks
        over the original candlestick chart. Default downsampling rule is:
        monthly for daily data, daily for hourly data, hourly for minute data,
        and minute for (sub-)second data.
        `superimpose` can also be a valid [Pandas offset string],
        such as `'5T'` or `'5min'`, in which case this frequency will be
        used to superimpose.
        Note, this only works for data with a datetime index.

        If `resample` is `True`, the OHLC data is resampled in a way that
        makes the upper number of candles for Bokeh to plot limited to 10_000.
        This may, in situations of overabundant data,
        improve plot's interactive performance and avoid browser's
        `Javascript Error: Maximum call stack size exceeded` or similar.
        Equity & dropdown curves and individual trades data is,
        likewise, [reasonably _aggregated_][TRADES_AGG].
        `resample` can also be a [Pandas offset string],
        such as `'5T'` or `'5min'`, in which case this frequency will be
        used to resample, overriding above numeric limitation.
        Note, all this only works for data with a datetime index.

        If `reverse_indicators` is `True`, the indicators below the OHLC chart
        are plotted in reverse order of declaration.

        [Pandas offset string]: \
            https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects

        [TRADES_AGG]: lib.html#backtesting.lib.TRADES_AGG

        If `show_legend` is `True`, the resulting plot graphs will contain
        labeled legends.

        If `open_browser` is `True`, the resulting `filename` will be
        opened in the default web browser.
        """
        if results is None:
            if self._results is None:
                raise RuntimeError('First issue `backtest.run()` to obtain results.')
            results = self._results

        return plot(
            results=results,
            df=self._data,
            indicators=results._strategy._indicators,
            filename=filename,
            plot_width=plot_width,
            plot_equity=plot_equity,
            plot_return=plot_return,
            plot_pl=plot_pl,
            plot_volume=plot_volume,
            plot_drawdown=plot_drawdown,
            plot_trades=plot_trades,
            smooth_equity=smooth_equity,
            relative_equity=relative_equity,
            superimpose=superimpose,
            resample=resample,
            reverse_indicators=reverse_indicators,
            show_legend=show_legend,
            open_browser=open_browser)


# NOTE: Don't put anything public below this __all__ list

__all__ = [getattr(v, '__name__', k)
           for k, v in globals().items()                        # export
           if ((callable(v) and getattr(v, '__module__', None) == __name__ or  # callables from this module; getattr for Python 3.9; # noqa: E501
                k.isupper()) and                                # or CONSTANTS
               not getattr(v, '__name__', k).startswith('_'))]  # neither marked internal

# NOTE: Don't put anything public below here. See above.
