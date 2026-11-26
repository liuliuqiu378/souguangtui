/* 轻量 SVG 图表库：无外部依赖，数据内联即可渲染，适配离线打开 */
(function (global) {
  const PALETTE = ["#4f46e5", "#0d9488", "#f59e0b", "#3b82f6", "#ec4899", "#8b5cf6", "#10b981", "#ef4444"];
  const NS = "http://www.w3.org/2000/svg";

  function el(tag, attrs, parent) {
    const n = document.createElementNS(NS, tag);
    for (const k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function fmt(v, type) {
    if (type === "pct") return (v * 100).toFixed(1) + "%";
    if (type === "pct0") return (v * 100).toFixed(0) + "%";
    if (type === "int") return v >= 10000 ? (v / 10000).toFixed(1) + "万" : Math.round(v).toLocaleString();
    if (type === "float") return Number(v).toFixed(2);
    return v;
  }
  function niceMax(v) {
    if (v <= 0) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    const n = v / p;
    const m = n <= 1 ? 1 : n <= 1.5 ? 1.5 : n <= 2 ? 2 : n <= 3 ? 3 : n <= 5 ? 5 : 10;
    return m * p;
  }
  function base(container, opts) {
    const host = typeof container === "string" ? document.querySelector(container) : container;
    if (!host) return null;
    const W = 820, H = opts.height || 300;
    host.innerHTML = "";
    const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.title || "chart" });
    host.appendChild(svg);
    if (opts.caption) {
      const c = document.createElement("div");
      c.className = "cap";
      c.innerHTML = opts.caption;
      host.appendChild(c);
    }
    return { svg, W, H };
  }
  function legend(host, items) {
    const d = document.createElement("div");
    d.className = "legend";
    d.innerHTML = items.map(i => `<span><i style="background:${i.color}"></i>${i.label}</span>`).join("");
    host.appendChild(d);
  }

  const Chart = {};

  /* 柱状图（默认竖向，可 horizontal） */
  Chart.bar = function (sel, data, opts = {}) {
    const host = typeof sel === "string" ? document.querySelector(sel) : sel;
    if (!host) return;
    const b = base(host, opts); if (!b) return;
    const { svg, W, H } = b;
    const horizontal = !!opts.horizontal;
    const valKey = opts.valueKey || "rate";
    const nameKey = opts.nameKey || "name";
    const color = opts.color || PALETTE[0];
    const type = opts.format || "float";
    const padL = horizontal ? 120 : 52, padR = 18, padT = 16, padB = horizontal ? 34 : 58;
    const iw = W - padL - padR, ih = H - padT - padB;
    const vals = data.map(d => +d[valKey]);
    const max = opts.max || niceMax(Math.max(...vals) * 1.12);

    for (let i = 0; i <= 4; i++) {
      const y = padT + ih - (ih * i) / 4;
      el("line", { x1: padL, x2: W - padR, y1: y, y2: y, stroke: "#eef1f6", "stroke-width": 1 }, svg);
      const t = el("text", { x: padL - 8, y: y + 4, "text-anchor": "end", "font-size": 10.5, fill: "#98a2b3" }, svg);
      t.textContent = fmt(max * i / 4, opts.axisFormat || type);
    }
    const n = data.length;
    if (horizontal) {
      const rowH = ih / n;
      data.forEach((d, i) => {
        const v = +d[valKey];
        const w = (v / max) * iw;
        const y = padT + i * rowH + rowH * 0.18;
        el("rect", { x: padL, y, width: Math.max(w, 1), height: rowH * 0.64, rx: 4, fill: color, opacity: .92 }, svg)
          .appendChild(el("title", {}, svg)).textContent = `${d[nameKey]}: ${fmt(v, type)}`;
        const nt = el("text", { x: padL - 10, y: y + rowH * 0.44, "text-anchor": "end", "font-size": 11.5, fill: "#475467" }, svg);
        nt.textContent = d[nameKey];
        const vt = el("text", { x: padL + w + 6, y: y + rowH * 0.44, "font-size": 10.5, fill: "#667085" }, svg);
        vt.textContent = fmt(v, opts.labelFormat || type);
      });
    } else {
      const bw = iw / n;
      data.forEach((d, i) => {
        const v = +d[valKey];
        const h = (v / max) * ih;
        const x = padL + i * bw + bw * 0.18;
        const rect = el("rect", { x, y: padT + ih - h, width: bw * 0.64, height: Math.max(h, 1), rx: 4, fill: color, opacity: .92 }, svg);
        const title = el("title", {}, rect);
        title.textContent = `${d[nameKey]}: ${fmt(v, type)}${d.n ? "（样本 " + d.n + "）" : ""}`;
        const vt = el("text", { x: x + bw * 0.32, y: padT + ih - h - 6, "text-anchor": "middle", "font-size": 10, fill: "#667085" }, svg);
        vt.textContent = fmt(v, opts.labelFormat || type);
        const nt = el("text", { x: x + bw * 0.32, y: H - padB + 16, "text-anchor": "middle", "font-size": 11, fill: "#475467" }, svg);
        nt.textContent = d[nameKey];
      });
      el("line", { x1: padL, x2: W - padR, y1: padT + ih, y2: padT + ih, stroke: "#d0d5dd" }, svg);
    }
    return b;
  };

  /* 分组柱状图 */
  Chart.groupBar = function (sel, data, opts = {}) {
    const host = typeof sel === "string" ? document.querySelector(sel) : sel;
    if (!host) return;
    const b = base(host, opts); if (!b) return;
    const { svg, W, H } = b;
    const keys = opts.keys; // [{key,label,color}]
    const padL = 52, padR = 18, padT = 18, padB = 48;
    const iw = W - padL - padR, ih = H - padT - padB;
    const all = data.flatMap(d => keys.map(k => +d[k.key]));
    const max = opts.max || niceMax(Math.max(...all));
    const type = opts.format || "float";
    for (let i = 0; i <= 4; i++) {
      const y = padT + ih - (ih * i) / 4;
      el("line", { x1: padL, x2: W - padR, y1: y, y2: y, stroke: "#eef1f6" }, svg);
      el("text", { x: padL - 8, y: y + 4, "text-anchor": "end", "font-size": 10.5, fill: "#98a2b3" }, svg).textContent = fmt(max * i / 4, type);
    }
    const gw = iw / data.length;
    const bw = Math.min(28, (gw * 0.7) / keys.length);
    data.forEach((d, i) => {
      const gx = padL + i * gw + gw / 2;
      keys.forEach((k, ki) => {
        const v = +d[k.key];
        const h = (v / max) * ih;
        const x = gx - (keys.length * bw) / 2 + ki * bw;
        const r = el("rect", { x, y: padT + ih - h, width: bw - 3, height: Math.max(h, 1), rx: 3, fill: k.color || PALETTE[ki] }, svg);
        el("title", {}, r).textContent = `${d.name} · ${k.label}: ${fmt(v, type)}`;
        if (data.length <= 7) {
          el("text", { x: x + (bw - 3) / 2, y: padT + ih - h - 5, "text-anchor": "middle", "font-size": 9.5, fill: "#667085" }, svg).textContent = fmt(v, type);
        }
      });
      el("text", { x: gx, y: H - padB + 18, "text-anchor": "middle", "font-size": 11.5, fill: "#475467" }, svg).textContent = d.name;
    });
    el("line", { x1: padL, x2: W - padR, y1: padT + ih, y2: padT + ih, stroke: "#d0d5dd" }, svg);
    legend(host, keys.map((k, i) => ({ label: k.label, color: k.color || PALETTE[i] })));
  };

  /* 折线图 */
  Chart.line = function (sel, data, opts = {}) {
    const host = typeof sel === "string" ? document.querySelector(sel) : sel;
    if (!host) return;
    const b = base(host, opts); if (!b) return;
    const { svg, W, H } = b;
    const keys = opts.keys; const xKey = opts.xKey || "epoch";
    const padL = 56, padR = 20, padT = 18, padB = 44;
    const iw = W - padL - padR, ih = H - padT - padB;
    const xs = data.map(d => +d[xKey]);
    const all = data.flatMap(d => keys.map(k => +d[k.key]));
    const min = opts.min != null ? opts.min : Math.min(...all) * 0.995;
    const max = opts.max != null ? opts.max : Math.max(...all) * 1.004;
    const span = max - min || 1;
    const X = v => padL + ((v - xs[0]) / (xs[xs.length - 1] - xs[0] || 1)) * iw;
    const Y = v => padT + ih - ((v - min) / span) * ih;
    for (let i = 0; i <= 4; i++) {
      const y = padT + ih - (ih * i) / 4;
      el("line", { x1: padL, x2: W - padR, y1: y, y2: y, stroke: "#eef1f6" }, svg);
      el("text", { x: padL - 8, y: y + 4, "text-anchor": "end", "font-size": 10.5, fill: "#98a2b3" }, svg)
        .textContent = (min + (span * i) / 4).toFixed(3);
    }
    xs.forEach(x => {
      el("text", { x: X(x), y: H - padB + 18, "text-anchor": "middle", "font-size": 11, fill: "#667085" }, svg).textContent = x;
    });
    keys.forEach((k, ki) => {
      const c = k.color || PALETTE[ki];
      const pts = data.map(d => `${X(+d[xKey])},${Y(+d[k.key])}`).join(" ");
      el("polyline", { points: pts, fill: "none", stroke: c, "stroke-width": 2.4, "stroke-linejoin": "round" }, svg);
      data.forEach(d => {
        const cx = X(+d[xKey]), cy = Y(+d[k.key]);
        const dot = el("circle", { cx, cy, r: 3.5, fill: "#fff", stroke: c, "stroke-width": 2 }, svg);
        el("title", {}, dot).textContent = `${k.label} · ${d[xKey]}: ${(+d[k.key]).toFixed(4)}`;
      });
    });
    legend(host, keys.map((k, i) => ({ label: k.label, color: k.color || PALETTE[i] })));
  };

  /* 环形图 */
  Chart.donut = function (sel, data, opts = {}) {
    const host = typeof sel === "string" ? document.querySelector(sel) : sel;
    if (!host) return;
    const b = base(host, opts); if (!b) return;
    const { svg, W, H } = b;
    const key = opts.valueKey || "value";
    const total = data.reduce((s, d) => s + (+d[key]), 0);
    const cx = W / 2, cy = H / 2, r = Math.min(H, W) * 0.31, r2 = r * 0.62;
    let ang = -Math.PI / 2;
    data.forEach((d, i) => {
      const v = +d[key];
      const a = (v / total) * Math.PI * 2;
      const p = (rr, aa) => [cx + rr * Math.cos(aa), cy + rr * Math.sin(aa)];
      const [x1, y1] = p(r, ang), [x2, y2] = p(r, ang + a);
      const [x3, y3] = p(r2, ang + a), [x4, y4] = p(r2, ang);
      const large = a > Math.PI ? 1 : 0;
      const path = el("path", {
        d: `M ${x1} ${y1} A ${r} ${r} 0 ${large} 1 ${x2} ${y2} L ${x3} ${y3} A ${r2} ${r2} 0 ${large} 0 ${x4} ${y4} Z`,
        fill: PALETTE[i % PALETTE.length], opacity: .92
      }, svg);
      el("title", {}, path).textContent = `${d.name}: ${fmt(v, opts.format || "int")} (${(v / total * 100).toFixed(1)}%)`;
      if (v / total > 0.07) {
        const [mx, my] = p((r + r2) / 2, ang + a / 2);
        el("text", { x: mx, y: my + 4, "text-anchor": "middle", "font-size": 10.5, fill: "#fff", "font-weight": 600 }, svg).textContent = (v / total * 100).toFixed(0) + "%";
      }
      ang += a;
    });
    el("text", { x: cx, y: cy - 2, "text-anchor": "middle", "font-size": 20, "font-weight": 700, fill: "#111827" }, svg)
      .textContent = fmt(total, opts.centerFormat || "int");
    el("text", { x: cx, y: cy + 18, "text-anchor": "middle", "font-size": 11, fill: "#98a2b3" }, svg).textContent = opts.centerLabel || "合计";
    legend(host, data.map((d, i) => ({ label: d.name, color: PALETTE[i % PALETTE.length] })));
  };

  /* 漏斗图 */
  Chart.funnel = function (sel, data, opts = {}) {
    const host = typeof sel === "string" ? document.querySelector(sel) : sel;
    if (!host) return;
    const b = base(host, opts); if (!b) return;
    const { svg, W, H } = b;
    const top = data[0].value;
    const padT = 12, gap = 8;
    const bh = (H - padT * 2 - gap * (data.length - 1)) / data.length;
    data.forEach((d, i) => {
      const w = (d.value / top) * (W - 180);
      const x = (W - 180 - w) / 2 + 10;
      const y = padT + i * (bh + gap);
      const rect = el("rect", { x, y, width: Math.max(w, 2), height: bh, rx: 5, fill: PALETTE[i % 3 === 0 ? 0 : i % 3 === 1 ? 1 : 3], opacity: .88 }, svg);
      el("title", {}, rect).textContent = `${d.name}: ${d.value.toLocaleString()}`;
      el("text", { x: W - 160, y: y + bh / 2 + 4, "font-size": 12, fill: "#475467" }, svg)
        .textContent = `${d.name} ${d.value.toLocaleString()}`;
      el("text", { x: W - 160, y: y + bh / 2 + 4, "font-size": 12, fill: "#475467" }, svg);
      el("text", { x: W - 12, y: y + bh / 2 + 4, "text-anchor": "end", "font-size": 12, fill: "#111827", "font-weight": 600 }, svg)
        .textContent = i === 0 ? "100%" : (d.value / top * 100).toFixed(1) + "%";
      el("text", { x: x + 12, y: y + bh / 2 + 4, "font-size": 12, fill: "#fff", "font-weight": 600 }, svg).textContent = d.name;
    });
  };

  /* 散点图 */
  Chart.scatter = function (sel, points, opts = {}) {
    const host = typeof sel === "string" ? document.querySelector(sel) : sel;
    if (!host) return;
    const b = base(host, opts); if (!b) return;
    const { svg, W, H } = b;
    const padL = 56, padR = 20, padT = 18, padB = 46;
    const iw = W - padL - padR, ih = H - padT - padB;
    const xs = points.map(p => p.x), ys = points.map(p => p.y);
    const xmin = opts.xmin != null ? opts.xmin : Math.min(...xs), xmax = opts.xmax != null ? opts.xmax : Math.max(...xs);
    const ymin = opts.ymin != null ? opts.ymin : Math.min(...ys), ymax = opts.ymax != null ? opts.ymax : Math.max(...ys);
    const X = v => padL + ((v - xmin) / (xmax - xmin || 1)) * iw;
    const Y = v => padT + ih - ((v - ymin) / (ymax - ymin || 1)) * ih;
    for (let i = 0; i <= 4; i++) {
      const y = padT + ih - (ih * i) / 4;
      el("line", { x1: padL, x2: W - padR, y1: y, y2: y, stroke: "#eef1f6" }, svg);
      el("text", { x: padL - 8, y: y + 4, "text-anchor": "end", "font-size": 10.5, fill: "#98a2b3" }, svg)
        .textContent = (ymin + ((ymax - ymin) * i) / 4).toFixed(2);
      el("text", { x: padL + (iw * i) / 4, y: H - padB + 18, "text-anchor": "middle", "font-size": 10.5, fill: "#98a2b3" }, svg)
        .textContent = (xmin + ((xmax - xmin) * i) / 4).toFixed(2);
    }
    // 对角线参考
    el("line", { x1: padL, y1: padT + ih, x2: W - padR, y2: padT, stroke: "#cbd5e1", "stroke-dasharray": "4 4" }, svg);
    points.forEach(p => {
      const c = el("circle", { cx: X(p.x), cy: Y(p.y), r: 3, fill: opts.color || PALETTE[0], opacity: .55 }, svg);
      if (p.label) el("title", {}, c).textContent = p.label;
    });
    if (opts.note) {
      el("text", { x: padL, y: 12, "font-size": 11, fill: "#98a2b3" }, svg).textContent = opts.note;
    }
  };

  /* 水平进度条列表（适合展示指标对比） */
  Chart.bars = function (sel, data, opts = {}) {
    return Chart.bar(sel, data, Object.assign({ horizontal: true, height: 40 + data.length * 34 }, opts));
  };

  global.Chart = Chart;
})(window);
