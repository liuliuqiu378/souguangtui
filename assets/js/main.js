/* 站点公共脚本：侧边栏、进度、代码复制 */
(function () {
  const NAV = [
    {
      group: "开始",
      items: [
        { href: "index.html", n: "00", label: "学习总览与 8 周计划" }
      ]
    },
    {
      group: "第一部分 · 建立认知",
      items: [
        { href: "basics.html", n: "01", label: "搜广推全景 / 概念 / 数据" }
      ]
    },
    {
      group: "第二部分 · 核心模型",
      items: [
        { href: "models.html", n: "02", label: "召回与排序模型演进" }
      ]
    },
    {
      group: "第三部分 · 工程落地",
      items: [
        { href: "system.html", n: "03", label: "系统工程与踩坑手册" }
      ]
    },
    {
      group: "第四部分 · 实战项目",
      items: [
        { href: "project-1.html", n: "P1", label: "短视频推荐系统（全链路）" },
        { href: "project-2.html", n: "P2", label: "政务「政策找人」" }
      ]
    },
    {
      group: "拓展专题",
      items: [
        { href: "industry.html", n: "T1", label: "行业与商业全景" },
        { href: "advanced.html", n: "T2", label: "进阶模型专题" }
      ]
    },
    {
      group: "第五部分 · 冲刺",
      items: [
        { href: "interview.html", n: "04", label: "面试拷问与术语速查" }
      ]
    }
  ];

  const page = document.body.dataset.page || "index";
  const title = document.body.dataset.title || "搜广推教程";

  const sidebar = document.getElementById("sidebar");
  if (sidebar) {
    sidebar.innerHTML = `
      <a class="brand" href="index.html" style="text-decoration:none">
        <span class="brand-mark">推</span>
        <span><b>搜广推：从 0 到能干活</b><span>SEARCH · ADS · RECSYS</span></span>
      </a>
      ${NAV.map(g => `
        <div class="nav-group">${g.group}</div>
        <nav>
          ${g.items.map(it => `
            <a href="${it.href}" class="${it.href.replace('.html','') === page ? 'active' : ''}">
              <span class="num">${it.n}</span><span>${it.label}</span>
            </a>`).join("")}
        </nav>`).join("")}
      <div class="side-foot">
        9 章 · 2 个可演示项目 · 2 个拓展专题<br>
        全部数据为仿真生成，脚本可复现。<br>
        <a href="interview.html#terms" style="color:#a5b4fc">术语不认识？查术语速查 →</a>
      </div>`;
  }

  const crumb = document.getElementById("crumb");
  if (crumb) crumb.innerHTML = `<b>${title}</b>`;

  const progress = document.getElementById("progress-bar");
  if (progress) {
    const onScroll = () => {
      const h = document.documentElement;
      const p = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
      progress.style.width = Math.min(100, p * 100) + "%";
    };
    document.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
  }

  const btn = document.getElementById("menu-btn");
  if (btn && sidebar) {
    const backdrop = document.createElement("div");
    backdrop.className = "sidebar-backdrop";
    document.body.appendChild(backdrop);
    const close = () => {
      sidebar.classList.remove("open");
      backdrop.classList.remove("show");
      document.body.classList.remove("nav-locked");
    };
    btn.addEventListener("click", () => {
      const open = sidebar.classList.toggle("open");
      backdrop.classList.toggle("show", open);
      document.body.classList.toggle("nav-locked", open);
    });
    backdrop.addEventListener("click", close);
    sidebar.addEventListener("click", e => {
      if (e.target.tagName === "A") close();
    });
    document.addEventListener("keydown", e => {
      if (e.key === "Escape") close();
    });
    window.addEventListener("resize", () => {
      if (window.innerWidth > 1080) close();
    });
  }

  const file = (location.pathname.split("/").pop() || "index.html");
  document.querySelectorAll("pre").forEach(pre => {
    const wrap = document.createElement("div");
    wrap.style.position = "relative";
    pre.parentNode.insertBefore(wrap, pre);
    wrap.appendChild(pre);
    const b = document.createElement("button");
    b.textContent = "复制";
    b.style.cssText = "position:absolute;right:10px;top:8px;font-size:11px;padding:3px 9px;border-radius:6px;border:1px solid #334155;background:#1e293b;color:#cbd5e1;cursor:pointer";
    b.onclick = () => {
      navigator.clipboard.writeText(pre.innerText).then(() => {
        b.textContent = "已复制";
        setTimeout(() => (b.textContent = "复制"), 1400);
      });
    };
    wrap.appendChild(b);
  });

  // 页面内锚点平滑滚动（考虑吸顶栏）
  document.querySelectorAll('a[href^="#"]').forEach(a => {
    a.addEventListener("click", e => {
      const id = a.getAttribute("href").slice(1);
      const target = document.getElementById(id);
      if (target) {
        e.preventDefault();
        window.scrollTo({ top: target.getBoundingClientRect().top + window.scrollY - 72, behavior: "smooth" });
      }
    });
  });

  if (file === "index.html" || file === "") document.querySelectorAll(".sidebar a").forEach(a => {
    if (a.getAttribute("href") === "index.html") a.classList.add("active");
  });

  // 本页目录高亮（滚动到哪个 h2，就点亮对应目录项）
  const toc = document.querySelector(".toc");
  if (toc) {
    const links = Array.from(toc.querySelectorAll('a[href^="#"]'));
    const heads = links.map(a => document.getElementById(a.getAttribute("href").slice(1))).filter(Boolean);
    const spy = () => {
      let cur = null;
      for (const h of heads) {
        if (h.getBoundingClientRect().top <= 120) cur = h.id;
      }
      links.forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + cur));
    };
    document.addEventListener("scroll", spy, { passive: true });
    spy();
  }

  // 回到顶部
  const top = document.createElement("button");
  top.className = "to-top";
  top.textContent = "↑";
  top.title = "回到顶部";
  top.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
  document.body.appendChild(top);
  const toggleTop = () => top.classList.toggle("show", window.scrollY > 600);
  document.addEventListener("scroll", toggleTop, { passive: true });
  toggleTop();

  // 打印/导出 PDF 时自动展开所有问答，打印完再恢复原状
  let openedForPrint = [];
  window.addEventListener("beforeprint", () => {
    openedForPrint = Array.from(document.querySelectorAll("details:not([open])"));
    openedForPrint.forEach(d => { d.open = true; });
  });
  window.addEventListener("afterprint", () => {
    openedForPrint.forEach(d => { d.open = false; });
    openedForPrint = [];
  });
})();
