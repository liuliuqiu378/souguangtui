(function () {
  var A = window.DATA_A;
  var B = window.DATA_B;
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ok = style.getPropertyValue('--ok').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();

  function init(id) {
    var el = document.getElementById(id);
    if (!el || !window.echarts) return null;
    var c = echarts.init(el, null, { renderer: 'svg' });
    window.addEventListener('resize', function () { c.resize(); });
    return c;
  }
  function axis(name) {
    return {
      name: name, nameTextStyle: { color: muted, fontSize: 11 },
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    };
  }
  function base(extra) {
    return Object.assign({
      animation: false,
      backgroundColor: 'transparent',
      textStyle: { fontFamily: 'Work Sans, PingFang SC, Microsoft YaHei, sans-serif' },
      tooltip: { trigger: 'axis', appendToBody: true, confine: true },
      grid: { left: 56, right: 30, top: 46, bottom: 42 }
    }, extra || {});
  }
  function shortTs(ts) { return ts.map(function (t) { return t.slice(5, 16); }); }

  // ================= 项目一：7 天指标总览 =================
  if (A) {
    var c1 = init('chart-a-series');
    if (c1) {
      var s = A.series;
      c1.setOption(base({
        tooltip: { trigger: 'axis', appendToBody: true, confine: true },
        legend: { top: 4, textStyle: { color: muted, fontSize: 11 }, itemWidth: 14, itemHeight: 8 },
        grid: { left: 60, right: 62, top: 48, bottom: 52 },
        xAxis: Object.assign(axis(''), {
          type: 'category', data: s.timestamps, boundaryGap: false,
          axisLabel: { color: muted, fontSize: 10, formatter: function (v) { return v.slice(5, 10); } }
        }),
        yAxis: [
          Object.assign(axis('QPS'), { type: 'value', scale: true }),
          Object.assign(axis('P99 (ms)'), { type: 'value', scale: true, splitLine: { show: false } })
        ],
        dataZoom: [{ type: 'inside', start: 0, end: 100 }, {
          type: 'slider', height: 16, bottom: 8, borderColor: rule,
          fillerColor: 'rgba(29,78,216,.10)', handleStyle: { color: accent }
        }],
        series: [
          {
            name: '订单 QPS', type: 'line', showSymbol: false, smooth: true,
            lineStyle: { width: 1.6, color: accent }, itemStyle: { color: accent },
            areaStyle: { color: 'rgba(29,78,216,.06)' }, data: s.order_qps,
            markArea: {
              silent: true, itemStyle: { color: 'rgba(217,72,15,.10)' },
              data: s.bands.map(function (b) { return [{ xAxis: b.start }, { xAxis: b.end }]; })
            }
          },
          {
            name: '订单 P99', type: 'line', yAxisIndex: 1, showSymbol: false,
            lineStyle: { width: 1.2, color: accent2, opacity: .9 },
            itemStyle: { color: accent2 }, data: s.order_lat
          }
        ]
      }));
    }

    // 方法对比
    var c2 = init('chart-a-methods');
    if (c2) {
      var names = Object.keys(A.methods);
      var f1 = names.map(function (n) { return +(A.methods[n].f1 * 100).toFixed(1); });
      var pa = names.map(function (n) { return +(A.methods[n].pa_f1 * 100).toFixed(1); });
      var fa = names.map(function (n) { return A.methods[n].false_alarms_per_day; });
      c2.setOption(base({
        legend: { top: 2, textStyle: { color: muted, fontSize: 11 }, itemWidth: 14, itemHeight: 8 },
        grid: { left: 50, right: 56, top: 44, bottom: 34 },
        xAxis: Object.assign(axis(''), { type: 'category', data: names }),
        yAxis: [
          Object.assign(axis('F1 (%)'), { type: 'value', max: 100 }),
          Object.assign(axis('误报/天'), { type: 'value', splitLine: { show: false } })
        ],
        series: [
          { name: '点级 F1', type: 'bar', barWidth: 18, itemStyle: { color: accent, borderRadius: [3, 3, 0, 0] }, data: f1 },
          { name: '区间 F1 (PA)', type: 'bar', barWidth: 18, itemStyle: { color: accent2, borderRadius: [3, 3, 0, 0] }, data: pa },
          {
            name: '误报/天', type: 'line', yAxisIndex: 1, symbol: 'circle', symbolSize: 7,
            lineStyle: { color: ok, width: 1.4 }, itemStyle: { color: ok }, data: fa
          }
        ]
      }));
    }

    // 告警漏斗
    var c3 = init('chart-a-funnel');
    if (c3) {
      c3.setOption(base({
        tooltip: { trigger: 'item', appendToBody: true, confine: true, formatter: '{b}: {c}' },
        series: [{
          type: 'funnel', left: '8%', right: '8%', top: 24, bottom: 18,
          minSize: '22%', maxSize: '100%', sort: 'descending', gap: 3,
          label: { color: '#fff', fontSize: 12, formatter: '{b}\n{c}' },
          itemStyle: { borderWidth: 0 },
          data: A.alert_funnel.map(function (d, i) {
            return { name: d.stage, value: d.count, itemStyle: { color: [accent, '#3b6bea', '#6b8ef5', ok][i] } };
          })
        }]
      }));
    }

    // AE 误差
    var c4 = init('chart-a-ae');
    if (c4) {
      var ae = A.ae_curve;
      c4.setOption(base({
        tooltip: { trigger: 'axis', appendToBody: true, confine: true },
        grid: { left: 56, right: 26, top: 34, bottom: 40 },
        xAxis: Object.assign(axis(''), {
          type: 'category', data: ae.timestamps, boundaryGap: false,
          axisLabel: { color: muted, fontSize: 10, formatter: function (v) { return v.slice(5, 10); } }
        }),
        yAxis: Object.assign(axis('重建误差'), { type: 'value', scale: true }),
        series: [{
          name: '重建误差', type: 'line', showSymbol: false, data: ae.score,
          lineStyle: { color: accent, width: 1.2 }, itemStyle: { color: accent },
          areaStyle: { color: 'rgba(29,78,216,.08)' },
          markLine: {
            silent: true, symbol: 'none',
            lineStyle: { color: accent2, type: 'dashed', width: 1.2 },
            label: { color: accent2, fontSize: 10, formatter: '阈值 ' + ae.threshold },
            data: [{ yAxis: ae.threshold }]
          }
        }]
      }));
    }

    // 日志模板
    var c5 = init('chart-a-logs');
    if (c5) {
      var rows = A.log_top.rows.slice().reverse();
      c5.setOption(base({
        tooltip: {
          trigger: 'axis', appendToBody: true, confine: true, axisPointer: { type: 'shadow' },
          formatter: function (p) { return p[0].name + '<br>' + p[0].value + ' 条'; }
        },
        grid: { left: 300, right: 40, top: 16, bottom: 26 },
        xAxis: Object.assign(axis('日志条数'), { type: 'value' }),
        yAxis: Object.assign(axis(''), {
          type: 'category', data: rows.map(function (r) { return r.tpl.slice(0, 38); }),
          axisLabel: { color: muted, fontSize: 10, width: 280, overflow: 'truncate' }
        }),
        series: [{
          type: 'bar', barWidth: 12,
          data: rows.map(function (r) {
            return { value: r.count, itemStyle: { color: r.level === 'ERROR' ? accent2 : (r.level === 'WARN' ? '#e8a33d' : accent), borderRadius: [0, 3, 3, 0] } };
          })
        }]
      }));
    }

    // 容量预测 A
    var c6 = init('chart-a-cap');
    if (c6) {
      var cap = A.capacity;
      var histX = shortTs(cap.history_ts);
      var fcX = shortTs(cap.forecast_ts);
      var allX = histX.concat(fcX);
      var hist = cap.history.concat(new Array(fcX.length).fill(null));
      var fc = new Array(histX.length - 1).fill(null).concat([cap.history[cap.history.length - 1]]).concat(cap.forecast);
      var lo = new Array(histX.length - 1).fill(null).concat([cap.history[cap.history.length - 1]]).concat(cap.lower);
      var band = new Array(histX.length - 1).fill(null).concat([0]).concat(cap.upper.map(function (v, i) { return +(v - cap.lower[i]).toFixed(1); }));
      c6.setOption(base({
        legend: { top: 2, textStyle: { color: muted, fontSize: 11 }, itemWidth: 14, itemHeight: 8 },
        grid: { left: 54, right: 22, top: 44, bottom: 40 },
        xAxis: Object.assign(axis(''), { type: 'category', data: allX, boundaryGap: false,
          axisLabel: { color: muted, fontSize: 10, interval: Math.floor(allX.length / 8) } }),
        yAxis: Object.assign(axis('QPS'), { type: 'value', scale: true }),
        series: [
          { name: '实际', type: 'line', showSymbol: false, data: hist, lineStyle: { color: accent, width: 1.8 }, itemStyle: { color: accent } },
          { name: '预测', type: 'line', showSymbol: false, data: fc, lineStyle: { color: accent2, width: 1.8, type: 'dashed' }, itemStyle: { color: accent2 } },
          { name: '95% 置信下界', type: 'line', data: lo, stack: 'ci', showSymbol: false, lineStyle: { opacity: 0 }, areaStyle: { opacity: 0 }, tooltip: { show: false } },
          { name: '95% 置信区间', type: 'line', data: band, stack: 'ci', showSymbol: false, lineStyle: { opacity: 0 }, areaStyle: { color: 'rgba(217,72,15,.13)' }, tooltip: { show: false } },
          {
            name: '扩容阈值', type: 'line', data: new Array(allX.length).fill(cap.capacity), showSymbol: false,
            lineStyle: { color: muted, type: 'dotted', width: 1 }, itemStyle: { color: muted }
          }
        ]
      }));
    }
  }

  // ================= 项目二：拓扑 =================
  if (B) {
    var g = init('chart-b-topo');
    if (g) {
      var t = B.topology;
      var palette = { '业务系统': accent, '接入层': '#7c3aed', '计算节点': '#0b8f5a', '中间件': accent2, '基础设施': '#64748b' };
      g.setOption(base({
        tooltip: {
          appendToBody: true, confine: true,
          formatter: function (p) {
            if (p.dataType === 'edge') return '';
            return '<b>' + p.data.cn + '</b><br>' + p.data.name + '<br>类别：' + p.data.category;
          }
        },
        legend: { top: 0, textStyle: { color: muted, fontSize: 11 }, itemWidth: 10, itemHeight: 10 },
        series: [{
          type: 'graph', layout: 'force', roam: true, draggable: true,
          force: { repulsion: 260, edgeLength: [50, 110], gravity: 0.08 },
          categories: t.categories.map(function (c) { return { name: c, itemStyle: { color: palette[c] || accent } }; }),
          label: { show: true, position: 'right', fontSize: 10, color: ink, formatter: function (p) { return p.data.cn; } },
          lineStyle: { color: '#c3ccdb', width: 1, curveness: 0.08, opacity: .8 },
          emphasis: { focus: 'adjacency', lineStyle: { width: 2.4, color: accent } },
          data: t.nodes.map(function (n) {
            return {
              name: n.name, cn: n.cn, category: n.category,
              symbolSize: Math.max(10, Math.min(30, 10 + n.value * 0.7)),
              value: n.value
            };
          }),
          edges: t.links
        }]
      }));
    }

    // RCA 对比
    var r = init('chart-b-rca');
    if (r) {
      r.setOption(base({
        legend: { top: 2, textStyle: { color: muted, fontSize: 11 }, itemWidth: 14, itemHeight: 8 },
        grid: { left: 50, right: 24, top: 44, bottom: 34 },
        xAxis: Object.assign(axis(''), { type: 'category', data: B.rca.map(function (d) { return d.method; }) }),
        yAxis: Object.assign(axis('准确率 (%)'), { type: 'value', max: 100 }),
        series: [
          { name: 'Top-1', type: 'bar', barWidth: 20, itemStyle: { color: accent, borderRadius: [3, 3, 0, 0] },
            label: { show: true, position: 'top', fontSize: 10, color: muted, formatter: '{c}%' },
            data: B.rca.map(function (d) { return Math.round(d.top1 * 100); }) },
          { name: 'Top-3', type: 'bar', barWidth: 20, itemStyle: { color: '#9db4ea', borderRadius: [3, 3, 0, 0] },
            label: { show: true, position: 'top', fontSize: 10, color: muted, formatter: '{c}%' },
            data: B.rca.map(function (d) { return Math.round(d.top3 * 100); }) }
        ]
      }));
    }

    // 容量预测 B
    var cb = init('chart-b-cap');
    if (cb) {
      var c = B.capacity.ywtb;
      var hx = shortTs(c.history_ts);
      var fx = shortTs(c.forecast_ts);
      var ax = hx.concat(fx);
      var hs = c.history.concat(new Array(fx.length).fill(null));
      var fs = new Array(hx.length - 1).fill(null).concat([c.history[c.history.length - 1]]).concat(c.forecast);
      var up = new Array(hx.length - 1).fill(null).concat([c.history[c.history.length - 1]]).concat(c.upper);
      var lo = new Array(hx.length - 1).fill(null).concat([c.history[c.history.length - 1]]).concat(c.lower);
      cb.setOption(base({
        legend: { top: 2, textStyle: { color: muted, fontSize: 11 }, itemWidth: 14, itemHeight: 8 },
        grid: { left: 56, right: 24, top: 44, bottom: 40 },
        xAxis: Object.assign(axis(''), { type: 'category', data: ax, boundaryGap: false,
          axisLabel: { color: muted, fontSize: 10, interval: Math.floor(ax.length / 7) } }),
        yAxis: Object.assign(axis('QPS'), { type: 'value', scale: true }),
        series: [
          { name: '历史', type: 'line', showSymbol: false, data: hs, lineStyle: { color: accent, width: 1.6 }, itemStyle: { color: accent } },
          { name: '未来 7 天预测', type: 'line', showSymbol: false, data: fs, lineStyle: { color: accent2, width: 1.8, type: 'dashed' }, itemStyle: { color: accent2 } },
          { name: '上界', type: 'line', data: up, showSymbol: false, lineStyle: { opacity: 0 }, tooltip: { show: false } },
          { name: '下界', type: 'line', data: lo, showSymbol: false, lineStyle: { opacity: 0 }, tooltip: { show: false } },
          {
            name: '配额水位', type: 'line', data: new Array(ax.length).fill(c.quota), showSymbol: false,
            lineStyle: { color: accent2, type: 'dotted', width: 1 }, itemStyle: { color: accent2 }
          }
        ]
      }));
    }

    // 告警热力图
    var h = init('chart-b-heat');
    if (h) {
      var days = ['周一', '周二', '周三', '周四', '周五', '周六', '周日'];
      var hours = [];
      for (var i = 0; i < 24; i++) hours.push(i + '时');
      h.setOption(base({
        tooltip: {
          appendToBody: true, confine: true,
          formatter: function (p) { return days[p.value[1]] + ' ' + hours[p.value[0]] + '<br>告警 ' + p.value[2] + ' 条'; }
        },
        grid: { left: 54, right: 24, top: 26, bottom: 56 },
        xAxis: Object.assign(axis(''), { type: 'category', data: hours, axisLabel: { color: muted, fontSize: 10, interval: 1 } }),
        yAxis: Object.assign(axis(''), { type: 'category', data: days, axisLabel: { color: muted, fontSize: 11 } }),
        visualMap: {
          min: 0, max: 20, calculable: true, orient: 'horizontal', left: 'center', bottom: 4,
          itemWidth: 12, itemHeight: 60, textStyle: { color: muted, fontSize: 10 },
          inRange: { color: ['#eef2f9', '#b9c9e8', '#6d8fd6', accent, accent2] }
        },
        series: [{
          type: 'heatmap', data: B.heatmap,
          label: { show: false },
          itemStyle: { borderColor: bg2, borderWidth: 2 },
          emphasis: { itemStyle: { borderColor: ink, borderWidth: 1 } }
        }]
      }));
    }
  }

  // CSR 报告兜底：避免图表容器无数据时报错
  window.__CHARTS_READY__ = true;
})();
