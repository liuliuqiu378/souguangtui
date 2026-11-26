/* ============================================================
   站点脚本：侧边栏导航 / 阅读进度 / 本页目录 / 代码复制
   ============================================================ */
const NAV = [
  { group: "开始" },
  { no: "⌂", file: "index.html", name: "课程总览与学习路线" },
  { no: "0", file: "00-primer.html", name: "零基础补课：Python 与 ML 词汇" },
  { group: "基础篇" },
  { no: "01", file: "01-role.html", name: "岗位拆解与知识地图" },
  { no: "02", file: "02-basics.html", name: "时序数据与机器学习基础" },
  { no: "03", file: "03-data.html", name: "数据工程与数据质量治理" },
  { group: "核心能力" },
  { no: "04", file: "04-model.html", name: "特征工程与负荷预测建模" },
  { no: "05", file: "05-anomaly.html", name: "异常用能识别" },
  { no: "06", file: "06-deploy.html", name: "算法服务化与上线运维" },
  { group: "实战项目" },
  { no: "P1", file: "07-project-load.html", name: "项目一：短期负荷预测" },
  { no: "P2", file: "08-project-ems.html", name: "项目二：园区 EMS 异常与调度" },
  { group: "进阶专题" },
  { no: "B1", file: "10-power.html", name: "电力与能源行业基础" },
  { no: "B2", file: "11-timeseries.html", name: "时序数据工程进阶" },
  { no: "B3", file: "12-forecast-advanced.html", name: "预测算法进阶" },
  { no: "B4", file: "13-optimize.html", name: "优化调度与需求响应" },
  { no: "B5", file: "14-engineering.html", name: "算法工程实践" },
  { group: "冲刺" },
  { no: "09", file: "09-career.html", name: "求职表达与 90 天落地" }
];

function renderSidebar() {
  const box = document.getElementById("sidebar");
  if (!box) return;
  const cur = (location.pathname.split("/").pop() || "index.html").toLowerCase();
  let html = `
    <div class="brand">
      <div class="brand-mark">⚡</div>
      <div class="brand-name">能源数据 & 算法工程师<span class="brand-sub">ZERO → JOB-READY</span></div>
    </div>`;
  NAV.forEach((it) => {
    if (it.group) { html += `<div class="nav-group">${it.group}</div>`; return; }
    const active = it.file.toLowerCase() === cur ? " active" : "";
    html += `<a class="nav-item${active}" href="${it.file}">
      <span class="nav-num">${it.no}</span><span>${it.name}</span></a>`;
  });
  html += `<div class="sidebar-foot">
      <b>两个可演示项目</b><br>负荷预测 MAPE 1.78% · EMS 异常识别 F1 0.956<br><br>
      全部数据由脚本生成，指标可一键复现。
    </div>`;
  box.innerHTML = html;
}

function buildTOC() {
  const rail = document.getElementById("toc");
  if (!rail) return;
  const heads = [...document.querySelectorAll(".content h2")];
  if (!heads.length) { rail.style.display = "none"; return; }
  let html = `<div class="toc-title">本页目录</div>`;
  heads.forEach((h, i) => {
    if (!h.id) h.id = "sec-" + (i + 1);
    html += `<a href="#${h.id}" data-target="${h.id}">${h.textContent.replace(/^\d+\.\s*/, "")}</a>`;
  });
  rail.innerHTML = html;
  const links = [...rail.querySelectorAll("a")];
  const obs = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) {
        links.forEach((a) => a.classList.toggle("active", a.dataset.target === e.target.id));
      }
    });
  }, { rootMargin: "-15% 0px -70% 0px" });
  heads.forEach((h) => obs.observe(h));
}

function initCopy() {
  document.querySelectorAll(".code-box").forEach((box) => {
    const btn = box.querySelector(".copy");
    const pre = box.querySelector("pre");
    if (!btn || !pre) return;
    btn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(pre.innerText);
        btn.textContent = "已复制";
        setTimeout(() => (btn.textContent = "复制"), 1600);
      } catch (e) {
        btn.textContent = "复制失败";
      }
    });
  });
}

function initProgress() {
  const bar = document.getElementById("progress");
  if (!bar) return;
  const onScroll = () => {
    const h = document.documentElement;
    const total = h.scrollHeight - h.clientHeight;
    bar.style.width = (total > 0 ? (h.scrollTop / total) * 100 : 0) + "%";
  };
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();
}

function initMobile() {
  const btn = document.getElementById("menuBtn");
  if (btn) btn.addEventListener("click", () => document.body.classList.toggle("nav-open"));
  document.addEventListener("click", (e) => {
    if (window.innerWidth <= 980 && document.body.classList.contains("nav-open")) {
      if (!e.target.closest(".sidebar") && !e.target.closest("#menuBtn")) {
        document.body.classList.remove("nav-open");
      }
    }
  });
}

document.addEventListener("DOMContentLoaded", () => {
  renderSidebar();
  buildTOC();
  initCopy();
  initProgress();
  initMobile();
});
