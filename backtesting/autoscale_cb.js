if (!window._bt_scale_range) {
    // 如果全局还没有定义 _bt_scale_range 函数，则定义它
    window._bt_scale_range = function (range, min, max, pad) {
        "use strict";
        if (min !== Infinity && max !== -Infinity) {
            // 如果最大值和最小值有效
            pad = pad ? (max - min) * .03 : 0;  // 可选地添加 3% 的边距
            range.start = min - pad;  // 设置范围起点
            range.end = max + pad;    // 设置范围终点
        } else {
            // 如果数据非法，则报错
            console.error('backtesting: scale range error:', min, max, range);
        }
    };
}


clearTimeout(window._bt_autoscale_timeout);  // 清除之前设置的自动缩放延时器


window._bt_autoscale_timeout = setTimeout(function () {
    /**
     * @variable cb_obj `fig_ohlc.x_range` 图表的X轴范围对象
     * @variable source `ColumnDataSource` Bokeh的数据源
     * @variable ohlc_range `fig_ohlc.y_range` OHLC图的Y轴范围
     * @variable volume_range `fig_volume.y_range` 成交量图的Y轴范围
     */
    "use strict";

    // 计算当前X轴视图窗口内的索引范围 i 到 j
    let i = Math.max(Math.floor(cb_obj.start), 0),
        j = Math.min(Math.ceil(cb_obj.end), source.data['ohlc_high'].length);

    // 获取视图窗口内的最高价和最低价
    let max = Math.max.apply(null, source.data['ohlc_high'].slice(i, j)),
        min = Math.min.apply(null, source.data['ohlc_low'].slice(i, j));
    _bt_scale_range(ohlc_range, min, max, true);  // 调整OHLC Y轴范围，含3%边距

    if (volume_range) {
        // 获取视图窗口内的最大成交量
        max = Math.max.apply(null, source.data['Volume'].slice(i, j));
        _bt_scale_range(volume_range, 0, max * 1.03, false);  // 成交量图从0开始，无需边距
    }

}, 50);  // 延迟50毫秒后执行，避免频繁触发

