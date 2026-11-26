(function () {
  var PAGES = [
    { id: 'index', href: 'index.html', label: '首页 · 岗位与路线', idx: '00' },
    { id: 'basics', href: 'basics.html', label: '基础篇 · 概念与数据', idx: '01' },
    { id: 'advanced', href: 'advanced.html', label: '进阶篇 · LLM 与落地', idx: '02' },
    { id: 'projects', href: 'projects.html', label: '实战篇 · 两个项目', idx: '03' },
    { id: 'industry', href: 'industry.html', label: '行业落地 · 场景与选型', idx: '04' },
    { id: 'appendix', href: 'appendix.html', label: '附录 · FAQ 与题库', idx: '05' }
  ];

  var current = document.body.getAttribute('data-page') || 'index';
  var sidebar = document.getElementById('sidebar');
  var content = document.getElementById('content');

  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
  }

  if (sidebar) {
    sidebar.innerHTML = '';
    var brand = el('a', 'brand');
    brand.href = 'index.html';
    brand.style.textDecoration = 'none';
    brand.innerHTML = '<div class="brand-mark">AI</div>' +
      '<div class="brand-text"><strong>AIOps 实战手册</strong><span>从监控到智能运维</span></div>';
    sidebar.appendChild(brand);

    var nav = el('nav', 'page-nav');
    PAGES.forEach(function (p) {
      var a = el('a', p.id === current ? 'active' : '');
      a.href = p.href;
      a.innerHTML = '<span>' + p.label + '</span><span class="idx">' + p.idx + '</span>';
      nav.appendChild(a);
    });
    sidebar.appendChild(nav);

    // 自动生成当前页目录
    if (content) {
      var heads = content.querySelectorAll('h2');
      if (heads.length) {
        sidebar.appendChild(el('div', 'toc-title', '本页目录'));
        var toc = el('div', 'toc');
        heads.forEach(function (h, i) {
          if (!h.id) h.id = 'sec-' + i;
          var a = el('a', '');
          a.href = '#' + h.id;
          a.setAttribute('data-target', h.id);
          var num = h.querySelector('.num');
          var n = num ? num.textContent.replace(/^0+/, '') : '';
          if (num && !n) n = '0';
          a.textContent = (n ? n + ' · ' : '') +
            h.textContent.replace(num ? num.textContent : '', '').trim();
          toc.appendChild(a);
        });
        sidebar.appendChild(toc);
      }
    }

    sidebar.appendChild(el('div', 'sidebar-foot',
      '全部指标均由本地仿真脚本真实跑出<br>数据不含任何真实企业信息'));

    // 滚动高亮
    var links = sidebar.querySelectorAll('.toc a');
    var map = {};
    links.forEach(function (l) { map[l.getAttribute('data-target')] = l; });
    var obs = new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) {
          links.forEach(function (l) { l.classList.remove('active'); });
          var l = map[en.target.id];
          if (l) l.classList.add('active');
        }
      });
    }, { rootMargin: '-70px 0px -70% 0px' });
    content.querySelectorAll('h2[id]').forEach(function (h) { obs.observe(h); });
  }

  // 移动端菜单
  var btn = document.getElementById('menuBtn');
  if (btn) {
    btn.addEventListener('click', function () {
      sidebar.classList.toggle('open');
    });
  }

  // 阅读进度
  var bar = document.querySelector('.progress');
  window.addEventListener('scroll', function () {
    var h = document.documentElement;
    var p = h.scrollTop / (h.scrollHeight - h.clientHeight || 1);
    if (bar) bar.style.width = (p * 100).toFixed(1) + '%';
  });

  // Mermaid
  if (window.mermaid) {
    mermaid.initialize({ startOnLoad: true, theme: 'neutral', securityLevel: 'loose' });
  }
})();
