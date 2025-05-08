# 尝试从模块中导入版本号
try:
    from ._version import version as __version__
except ImportError:
    # 如果模块未安装，则版本号为 '?.?.?'
    __version__ = '?.?.?'  # 包未安装

# 导入一些内部库，F401 是 flake8 的忽略标记，表示 "imported but unused"
from . import lib  # noqa: F401
from ._plotting import set_bokeh_output  # noqa: F401
from .backtesting import Backtest, Strategy  # noqa: F401


# 定义一个可覆盖的 Pool 函数，用于并行优化
def Pool(processes=None, initializer=None, initargs=()):
    import multiprocessing as mp
    # 检查当前多进程启动方法是否为 'spawn'（例如在 Windows 上）
    if mp.get_start_method() == 'spawn':
        import warnings
        warnings.warn(
            "如果您想使用多进程优化并且 `multiprocessing.get_start_method() == 'spawn'`（例如在 Windows 上），"
            "请设置 `backtesting.Pool = multiprocessing.Pool` （或所需的上下文）"
            "并将 `bt.optimize()` 调用放在 `if __name__ == '__main__'` 的保护块后面。"
            "当前使用的是基于线程的并行处理，"
            "对于非 numpy / 非 GIL 释放型代码可能会稍慢一些。"
            "详见 https://github.com/kernc/backtesting.py/issues/1256",
            category=RuntimeWarning, stacklevel=3)
        # 使用 dummy.Pool（即线程池）作为替代方案
        from multiprocessing.dummy import Pool
        return Pool(processes, initializer, initargs)
    else:
        # 否则使用标准的多进程 Pool
        return mp.Pool(processes, initializer, initargs)
