[![](https://i.imgur.com/E8Kj69Y.png)](https://kernc.github.io/backtesting.py/)

Backtesting.py
==============

使用Python进行交易策略的回测。


Installation
------------

    $ pip install backtesting


Usage
-----
```python
from backtesting import Backtest, Strategy
from backtesting.lib import crossover

from backtesting.test import SMA, GOOG


class SmaCross(Strategy):
    def init(self):
        price = self.data.Close
        self.ma1 = self.I(SMA, price, 10)
        self.ma2 = self.I(SMA, price, 20)

    def next(self):
        if crossover(self.ma1, self.ma2):
            self.buy()
        elif crossover(self.ma2, self.ma1):
            self.sell()


bt = Backtest(GOOG, SmaCross, commission=.002,
              exclusive_orders=True)
stats = bt.run()
bt.plot()
```

Results in:

```text
开始                     2004-08-19 00:00:00
结束                       2013-03-01 00:00:00
持续时间                   3116天 00:00:00
持仓时间 [%]                       94.27
最终权益 [$]                     68935.12
最高权益 [$]                      68991.22
收益率 [%]                             589.35
买入并持有收益率 [%]                  703.46
年化收益率 [%]                       25.42
年化波动率 [%]                   38.43
复合年增长率 [%]                                16.80
夏普比率                             0.66
索提诺比率                            1.30
卡尔马比率                             0.77
Alpha [%]                              450.62
Beta                                     0.02
最大回撤 [%]                      -33.08
平均回撤 [%]                       -5.58
最长回撤周期      688天 00:00:00
平均回撤周期       41天 00:00:00
交易次数                                   93
胜率 [%]                            53.76
最佳交易 [%]                          57.12
最差交易 [%]                        -16.63
平均交易 [%]                           1.96
最长交易周期         121天 00:00:00
平均交易周期          32天 00:00:00
盈利因子                            2.13
期望收益 [%]                           6.91
SQN                                      1.78
凯利准则                        0.6134
_strategy              SmaCross(n1=10, n2=20)
_equity_curve                          Equ...
_trades                       Size  EntryB...
dtype: object

```
[![plot of trading simulation](./xRFNHfg.png)](https://kernc.github.io/backtesting.py/#example)




特性
--------
- 简单、有良好文档记录的API

- 极速执行
- 内置优化器
- 可组合的基础策略和实用工具库
- 指标库无关性
- 支持任何具有烛台数据的金融工具
- 详细的结果
- 交互式可视化

