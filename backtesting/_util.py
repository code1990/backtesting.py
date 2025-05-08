from __future__ import annotations

import os
import sys
import warnings
from contextlib import contextmanager
from functools import partial
from itertools import chain
from multiprocessing import resource_tracker as _mprt
from multiprocessing import shared_memory as _mpshm
from numbers import Number
from threading import Lock
from typing import Dict, List, Optional, Sequence, Union, cast

import numpy as np
import pandas as pd

# 尝试导入 tqdm 进度条库
try:
    from tqdm.auto import tqdm as _tqdm

    _tqdm = partial(_tqdm, leave=False)
except ImportError:
    def _tqdm(seq, **_):
        return seq


def try_(lazy_func, default=None, exception=Exception):
    """
    安全执行函数，捕获异常并返回默认值。

    参数:
        lazy_func (callable): 要执行的函数。
        default: 出现异常时的默认返回值。
        exception (Exception): 要捕获的异常类型。

    返回:
        执行结果或默认值。
    """
    try:
        return lazy_func()
    except exception:
        return default


@contextmanager
def patch(obj, attr, newvalue):
    """
    临时修改对象属性，并在退出上下文后恢复。

    示例:
        with patch(obj, 'attr', value):
            ...
    """
    had_attr = hasattr(obj, attr)
    orig_value = getattr(obj, attr, None)
    setattr(obj, attr, newvalue)
    try:
        yield
    finally:
        if had_attr:
            setattr(obj, attr, orig_value)
        else:
            delattr(obj, attr)


def _as_str(value) -> str:
    """
    将任意对象转换为字符串表示形式。
    """
    if isinstance(value, (Number, str)):
        return str(value)
    if isinstance(value, pd.DataFrame):
        return 'df'
    name = str(getattr(value, 'name', '') or '')
    if name in ('Open', 'High', 'Low', 'Close', 'Volume'):
        return name[:1]
    if callable(value):
        name = getattr(value, '__name__', value.__class__.__name__).replace('<lambda>', 'λ')
    if len(name) > 10:
        name = name[:9] + '…'
    return name


def _as_list(value) -> List:
    """
    将非字符串序列转换为列表。
    """
    if isinstance(value, Sequence) and not isinstance(value, str):
        return list(value)
    return [value]


def _batch(seq):
    """
    将序列分成多个小批次处理（用于多进程优化）。
    """
    n = np.clip(int(len(seq) // (os.cpu_count() or 1)), 1, 300)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


def _data_period(index) -> Union[pd.Timedelta, Number]:
    """
    计算数据的时间周期（如每根K线的时间间隔）。
    """
    values = pd.Series(index[-100:])
    return values.diff().dropna().median()


def _strategy_indicators(strategy):
    """
    获取策略中的所有指标。
    """
    return {attr: indicator
            for attr, indicator in strategy.__dict__.items()
            if isinstance(indicator, _Indicator)}.items()


def _indicator_warmup_nbars(strategy):
    """
    计算策略需要预热的 K 线数量。
    """
    if strategy is None:
        return 0
    nbars = max((np.isnan(indicator.astype(float)).argmin(axis=-1).max()
                 for _, indicator in _strategy_indicators(strategy)
                 if not indicator._opts['scatter']), default=0)
    return nbars


class _Array(np.ndarray):
    """
    扩展的 ndarray 类，支持 .name 和其他自定义属性。
    """

    def __new__(cls, array, *, name=None, **kwargs):
        obj = np.asarray(array).view(cls)
        obj.name = name or array.name
        obj._opts = kwargs
        return obj

    def __array_finalize__(self, obj):
        if obj is not None:
            self.name = getattr(obj, 'name', '')
            self._opts = getattr(obj, '_opts', {})

    def __reduce__(self):
        value = super().__reduce__()
        return value[:2] + (value[2] + (self.__dict__,),)

    def __setstate__(self, state):
        self.__dict__.update(state[-1])
        super().__setstate__(state[:-1])

    def __bool__(self):
        try:
            return bool(self[-1])
        except IndexError:
            return super().__bool__()

    def __float__(self):
        try:
            return float(self[-1])
        except IndexError:
            return super().__float__()

    def to_series(self):
        warnings.warn(
            "`.to_series()` 已弃用。请使用 [.s](file://D:\dev\dev_123\backtesting.py\venv\Lib\site-packages\numpy\_core\include\numpy\random\libdivide.h#L377-L377) 来进行 pd.Series 转换")
        return self.s

    @property
    def s(self) -> pd.Series:
        """转换为 pd.Series"""
        values = np.atleast_2d(self)
        index = self._opts['index'][:values.shape[1]]
        return pd.Series(values[0], index=index, name=self.name)

    @property
    def df(self) -> pd.DataFrame:
        """转换为 pd.DataFrame"""
        values = np.atleast_2d(np.asarray(self))
        index = self._opts['index'][:values.shape[1]]
        df = pd.DataFrame(values.T, index=index, columns=[self.name] * len(values))
        return df


class _Indicator(_Array):
    """指示器类，继承自 _Array"""
    pass


class _Data:
    """
    数据访问器，提供对 OHLCV 数据的快速访问。
    返回的是 `np.ndarray` {:.6f}于性能原因，而不是 `pd.Series`。
    """

    def __init__(self, df: pd.DataFrame):
        self.__df = df
        self.__len = len(df)  # 当前长度
        self.__pip: Optional[float] = None
        self.__cache: Dict[str, _Array] = {}
        self.__arrays: Dict[str, _Array] = {}
        self._update()

    def __getitem__(self, item):
        return self.__get_array(item)

    def __getattr__(self, item):
        try:
            return self.__get_array(item)
        except KeyError:
            raise AttributeError(f"列 '{item}' 不存在") from None

    def _set_length(self, length):
        """设置当前长度"""
        self.__len = length
        self.__cache.clear()

    def _update(self):
        """更新内部缓存"""
        index = self.__df.index.copy()
        self.__arrays = {col: _Array(arr, index=index)
                         for col, arr in self.__df.items()}
        self.__arrays['__index'] = index  # 保留索引

    def __repr__(self):
        i = min(self.__len, len(self.__df)) - 1
        index = self.__arrays['__index'][i]
        items = ', '.join(f'{k}={v}' for k, v in self.__df.iloc[i].items())
        return f'<Data i={i} ({index}) {items}>'

    def __len__(self):
        return self.__len

    @property
    def df(self) -> pd.DataFrame:
        """获取原始 DataFrame 的切片"""
        return (self.__df.iloc[:self.__len]
                if self.__len < len(self.__df)
                else self.__df)

    @property
    def pip(self) -> float:
        """计算价格最小变动单位"""
        if self.__pip is None:
            self.__pip = float(10 ** -np.median([len(s.partition('.')[-1])
                                                 for s in self.__arrays['Close'].astype(str)]))
        return self.__pip

    def __get_array(self, key) -> _Array:
        """获取数组"""
        arr = self.__cache.get(key)
        if arr is None:
            arr = self.__cache[key] = cast(_Array, self.__arrays[key][:self.__len])
        return arr

    @property
    def Open(self) -> _Array:
        return self.__get_array('Open')

    @property
    def High(self) -> _Array:
        return self.__get_array('High')

    @property
    def Low(self) -> _Array:
        return self.__get_array('Low')

    @property
    def Close(self) -> _Array:
        return self.__get_array('Close')

    @property
    def Volume(self) -> _Array:
        return self.__get_array('Volume')

    @property
    def index(self) -> pd.DatetimeIndex:
        return self.__get_array('__index')

    # Pickling 支持
    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


if sys.version_info >= (3, 13):
    SharedMemory = _mpshm.SharedMemory
else:
    class SharedMemory(_mpshm.SharedMemory):
        """
        兼容旧版本 Python 的共享内存类。
        """
        __lock = Lock()

        def __init__(self, *args, track: bool = True, **kwargs):
            self._track = track
            if track:
                return super().__init__(*args, **kwargs)
            with self.__lock:
                with patch(_mprt, 'register', lambda *a, **kw: None):
                    super().__init__(*args, **kwargs)

        def unlink(self):
            """取消注册共享内存"""
            if _mpshm._USE_POSIX and self._name:
                _mpshm._posixshmem.shm_unlink(self._name)
                if self._track:
                    _mprt.unregister(self._name, "shared_memory")


class SharedMemoryManager:
    """
    共享内存管理器，基于 Python 标准库实现。
    """

    def __init__(self, create=False) -> None:
        self._shms: list[SharedMemory] = []
        self.__create = create

    def SharedMemory(self, *, name=None, create=False, size=0, track=True):
        shm = SharedMemory(name=name, create=create, size=size, track=track)
        shm._create = create
        self._shms.append(shm)
        return shm

    def __enter__(self):
        return self

    def __exit__(self, *args, **kwargs):
        for shm in self._shms:
            try:
                shm.close()
                if shm._create:
                    shm.unlink()
            except Exception:
                warnings.warn(f'无法取消链接共享内存 {shm.name!r}',
                              category=ResourceWarning, stacklevel=2)
                raise

    def arr2shm(self, vals):
        """将数组写入共享内存"""
        assert vals.ndim == 1, vals.shape
        shm = self.SharedMemory(size=vals.nbytes, create=True)
        buf = np.ndarray(vals.shape, dtype=vals.dtype.base, buffer=shm.buf)
        has_tz = getattr(vals.dtype, 'tz', None)
        buf[:] = vals.tz_localize(None) if has_tz else vals  # 拷贝数据到共享内存
        return shm.name, vals.shape, vals.dtype

    def df2shm(self, df):
        """将 DataFrame 写入共享内存"""
        return tuple((
            (column, *self.arr2shm(values))
            for column, values in chain([(self._DF_INDEX_COL, df.index)], df.items())
        ))

    @staticmethod
    def shm2s(shm, shape, dtype) -> pd.Series:
        """从共享内存读取 Series"""
        arr = np.ndarray(shape, dtype=dtype.base, buffer=shm.buf)
        arr.setflags(write=False)
        return pd.Series(arr, dtype=dtype)

    _DF_INDEX_COL = '__bt_index'

    @staticmethod
    def shm2df(data_shm):
        """从共享内存读取 DataFrame"""
        shm = [SharedMemory(name=name, create=False, track=False) for _, name, _, _ in data_shm]
        df = pd.DataFrame({
            col: SharedMemoryManager.shm2s(shm, shape, dtype)
            for shm, (col, _, shape, dtype) in zip(shm, data_shm)})
        df.set_index(SharedMemoryManager._DF_INDEX_COL, drop=True, inplace=True)
        df.index.name = None
        return df, shm
