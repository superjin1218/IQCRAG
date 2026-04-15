"""
Step 09d — 정적 사이트 조립.

output/site/
├── index.html              메인 (그래프/맵/히트맵 탭)
├── detail.html             필드 상세 (이웃 상관 테이블)
├── assets/
│   ├── style.css
│   ├── main.js
│   ├── detail.js
│   ├── data.json           필드 인덱스 (검색 + 카테고리)
│   └── neighbors/<fid>.json  필드별 이웃 상관 리스트 (on demand)
└── views/                  09a/b/c 가 만든 파일
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, resolve_path, log_line, update_step,
)

STEP = "09_visualize"


STYLE_CSS = r"""
:root {
  --bg-primary: #0a0a0a;
  --bg-surface: #121212;
  --bg-hover: #1a1a1a;
  --border: #2a2a2a;
  --text-mute: #6b6b6b;
  --text-base: #a8a8a8;
  --text-bright: #f0f0f0;
  --text-hi: #ffffff;
  --accent: #6bd1ff;
  --pos: #7fd488;
  --neg: #e06c6c;
  --sidebar-w: 300px;
}

* { box-sizing: border-box; }

html, body {
  margin: 0; padding: 0;
  height: 100%;
  background: var(--bg-primary);
  color: var(--text-base);
  font-family: 'JetBrains Mono', 'SF Mono', Consolas, monospace;
  font-size: 13px;
  -webkit-font-smoothing: antialiased;
}

a { color: var(--text-bright); text-decoration: none; }
a:hover { color: var(--text-hi); }

/* layout ------------------------------------------ */
.app {
  display: grid;
  grid-template-columns: var(--sidebar-w) 1fr;
  grid-template-rows: 56px 1fr 32px;
  grid-template-areas:
    "header header"
    "sidebar main"
    "footer footer";
  height: 100vh;
}

.header {
  grid-area: header;
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 24px;
  justify-content: space-between;
}
.header .brand {
  font-weight: 600;
  letter-spacing: -0.02em;
  color: var(--text-hi);
  font-size: 15px;
}
.header .meta {
  color: var(--text-mute);
  font-size: 11px;
  letter-spacing: 0.04em;
}

.sidebar {
  grid-area: sidebar;
  border-right: 1px solid var(--border);
  overflow-y: auto;
  padding: 20px 16px 32px 16px;
}

.main {
  grid-area: main;
  overflow: hidden;
  position: relative;
}

.footer {
  grid-area: footer;
  border-top: 1px solid var(--border);
  display: flex;
  align-items: center;
  padding: 0 24px;
  color: var(--text-mute);
  font-size: 10px;
  letter-spacing: 0.04em;
}

/* sidebar sections -------------------------------- */
.section { margin-bottom: 24px; }
.section-title {
  color: var(--text-mute);
  font-size: 10px;
  letter-spacing: 0.12em;
  margin-bottom: 10px;
  text-transform: uppercase;
}

.stat-row {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
}
.stat-row .num {
  color: var(--text-bright);
  font-variant-numeric: tabular-nums;
}

/* view selector ----------------------------------- */
.view-list { list-style: none; margin: 0; padding: 0; }
.view-list li {
  padding: 8px 12px;
  margin: 2px 0;
  cursor: pointer;
  color: var(--text-base);
  border-left: 2px solid transparent;
  transition: all 0.15s ease;
}
.view-list li:hover { background: var(--bg-hover); color: var(--text-bright); }
.view-list li.active {
  color: var(--text-hi);
  border-left-color: var(--text-hi);
  background: var(--bg-hover);
}
.view-list li .view-id {
  color: var(--text-mute);
  font-size: 10px;
  margin-right: 8px;
}

/* search ----------------------------------------- */
.search-input {
  width: 100%;
  background: var(--bg-surface);
  color: var(--text-bright);
  border: 1px solid var(--border);
  padding: 8px 10px;
  font-family: inherit;
  font-size: 12px;
  outline: none;
  transition: border-color 0.15s ease;
}
.search-input:focus { border-color: var(--text-hi); }

.search-results {
  margin-top: 8px;
  max-height: 320px;
  overflow-y: auto;
}
.search-result {
  padding: 8px 10px;
  border-left: 2px solid transparent;
  cursor: pointer;
  transition: background 0.1s;
}
.search-result:hover {
  background: var(--bg-hover);
  border-left-color: var(--text-hi);
}
.search-result .sr-id {
  color: var(--text-bright);
  font-size: 11px;
  font-weight: 500;
  word-break: break-all;
}
.search-result .sr-meta {
  color: var(--text-mute);
  font-size: 10px;
  margin-top: 2px;
  display: flex;
  justify-content: space-between;
  gap: 8px;
}
.search-result .sr-meta .cat { color: var(--text-base); }
.search-result .sr-meta .num {
  color: var(--text-mute);
  font-variant-numeric: tabular-nums;
}
.search-result .sr-actions {
  margin-top: 6px;
  display: flex;
  gap: 8px;
  font-size: 10px;
}
.search-result .sr-actions a {
  color: var(--accent);
  border: 1px solid var(--border);
  padding: 2px 6px;
  cursor: pointer;
}
.search-result .sr-actions a:hover {
  background: var(--bg-hover);
  border-color: var(--accent);
}
.search-result .sr-stats {
  margin-top: 4px;
  display: flex;
  gap: 10px;
  font-size: 10px;
  font-variant-numeric: tabular-nums;
}
.search-result .sr-stat { color: var(--text-base); }
.search-result .sr-stat em {
  color: var(--text-mute);
  font-style: normal;
  margin-right: 2px;
}
.search-result .sr-stat.good { color: var(--pos); }
.search-result .sr-stat.good em { color: var(--pos); opacity: 0.7; }
.search-empty {
  color: var(--text-mute);
  font-size: 10px;
  padding: 8px 10px;
}

/* category filter -------------------------------- */
.category-list { list-style: none; margin: 0; padding: 0; }
.category-list li {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
}
.category-list li .count {
  color: var(--text-mute);
  font-variant-numeric: tabular-nums;
}

/* main view container ----------------------------- */
.view-container {
  position: absolute;
  inset: 0;
  background: var(--bg-primary);
}
.view-iframe {
  width: 100%;
  height: 100%;
  border: none;
  background: var(--bg-primary);
}
.view-image-wrap {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  padding: 0;
  cursor: grab;
  touch-action: none;
  user-select: none;
  -webkit-user-select: none;
}
.view-image-wrap.grabbing { cursor: grabbing; }
.view-image {
  max-width: 90%;
  max-height: 90%;
  object-fit: contain;
  transform-origin: center center;
  will-change: transform;
  pointer-events: none;
}

/* empty state --------------------------- */
.empty-state {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  flex-direction: column;
  color: var(--text-mute);
  font-size: 12px;
  gap: 12px;
}
.empty-state .dot {
  width: 6px; height: 6px;
  border-radius: 50%;
  background: var(--text-mute);
  animation: pulse 1.4s ease-in-out infinite;
}
@keyframes pulse {
  0%, 100% { opacity: 0.3; }
  50% { opacity: 1.0; }
}

/* scrollbar --------------------------------------- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); }
::-webkit-scrollbar-thumb:hover { background: var(--text-mute); }

/* ================================================= */
/*  detail.html                                      */
/* ================================================= */
.detail-app {
  min-height: 100vh;
  padding: 32px 48px 64px 48px;
  max-width: 1280px;
  margin: 0 auto;
}
.detail-header {
  border-bottom: 1px solid var(--border);
  padding-bottom: 16px;
  margin-bottom: 24px;
}
.detail-back {
  color: var(--text-mute);
  font-size: 11px;
  margin-bottom: 8px;
  display: inline-block;
}
.detail-back:hover { color: var(--text-bright); }
.detail-title {
  color: var(--text-hi);
  font-size: 24px;
  font-weight: 500;
  letter-spacing: -0.02em;
  word-break: break-all;
  margin: 4px 0 8px 0;
}
.detail-meta {
  color: var(--text-mute);
  font-size: 12px;
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.detail-meta .label { color: var(--text-mute); }
.detail-meta .val {
  color: var(--text-bright);
  margin-left: 4px;
  font-variant-numeric: tabular-nums;
}

.detail-controls {
  display: flex;
  gap: 12px;
  align-items: center;
  margin-bottom: 16px;
  flex-wrap: wrap;
}
.detail-controls .filter-input {
  flex: 1;
  min-width: 200px;
  background: var(--bg-surface);
  color: var(--text-bright);
  border: 1px solid var(--border);
  padding: 8px 12px;
  font-family: inherit;
  font-size: 12px;
  outline: none;
}
.detail-controls .filter-input:focus { border-color: var(--text-hi); }
.detail-controls .count-note {
  color: var(--text-mute);
  font-size: 11px;
  margin-left: auto;
}

table.neighbors {
  width: 100%;
  border-collapse: collapse;
  font-size: 12px;
}
table.neighbors thead th {
  text-align: left;
  color: var(--text-mute);
  font-size: 10px;
  font-weight: 500;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  padding: 10px 12px;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  user-select: none;
  white-space: nowrap;
}
table.neighbors thead th:hover { color: var(--text-bright); }
table.neighbors thead th.sorted { color: var(--text-hi); }
table.neighbors thead th .arrow {
  display: inline-block;
  margin-left: 4px;
  font-size: 9px;
  color: var(--accent);
}
table.neighbors tbody tr {
  border-bottom: 1px solid #141414;
  transition: background 0.1s;
}
table.neighbors tbody tr:hover { background: var(--bg-hover); }
table.neighbors tbody td {
  padding: 8px 12px;
  color: var(--text-base);
  font-variant-numeric: tabular-nums;
}
table.neighbors tbody td.fid {
  color: var(--text-bright);
  word-break: break-all;
  max-width: 260px;
}
table.neighbors tbody td.fid a {
  color: var(--text-bright);
}
table.neighbors tbody td.fid a:hover { color: var(--accent); }
table.neighbors tbody td.num-pos { color: var(--pos); }
table.neighbors tbody td.num-neg { color: var(--neg); }
table.neighbors tbody td .cat {
  color: var(--text-mute);
  font-size: 11px;
}
.detail-loading, .detail-error {
  padding: 40px;
  text-align: center;
  color: var(--text-mute);
}
.detail-error { color: var(--neg); }

/* ================================================= */
/*  GROUPS panel                                     */
/* ================================================= */
.groups-panel {
  position: absolute;
  inset: 0;
  overflow-y: auto;
  padding: 32px 40px 64px 40px;
}
.groups-loading {
  color: var(--text-mute);
  font-size: 12px;
  padding: 20px;
}
.groups-header { margin-bottom: 24px; }
.groups-title {
  color: var(--text-hi);
  font-size: 20px;
  font-weight: 500;
  letter-spacing: -0.01em;
  margin-bottom: 6px;
}
.groups-explain {
  color: var(--text-mute);
  font-size: 11px;
  margin-bottom: 16px;
}
.groups-export {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 16px;
  padding: 10px 14px;
  background: var(--bg-surface);
  border: 1px solid var(--border);
  font-size: 11px;
}
.groups-export .ex-label { color: var(--text-mute); }
.groups-export .ex-input {
  width: 60px;
  background: var(--bg-primary);
  color: var(--text-bright);
  border: 1px solid var(--border);
  padding: 4px 8px;
  font-family: inherit;
  font-size: 12px;
  font-variant-numeric: tabular-nums;
  text-align: center;
  outline: none;
}
.groups-export .ex-input:focus { border-color: var(--text-hi); }
.groups-export .ex-btn {
  margin-left: auto;
  background: var(--bg-primary);
  color: var(--accent);
  border: 1px solid var(--border);
  padding: 6px 14px;
  font-family: inherit;
  font-size: 11px;
  cursor: pointer;
  transition: all 0.15s ease;
}
.groups-export .ex-btn:hover {
  background: var(--bg-hover);
  border-color: var(--accent);
}
.groups-slider {
  background: var(--bg-surface);
  border: 1px solid var(--border);
  padding: 16px 20px 14px 20px;
  margin-bottom: 12px;
}
.groups-slider .slider-top {
  display: flex;
  align-items: baseline;
  gap: 12px;
  margin-bottom: 12px;
}
.groups-slider .slider-label {
  color: var(--text-mute);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.groups-slider .slider-value {
  color: var(--accent);
  font-size: 26px;
  font-weight: 500;
  font-variant-numeric: tabular-nums;
  min-width: 70px;
}
.groups-slider .slider-count {
  margin-left: auto;
  color: var(--text-mute);
  font-size: 11px;
}
.groups-slider .slider-count b {
  color: var(--text-hi);
  font-weight: 500;
  margin-right: 4px;
  font-variant-numeric: tabular-nums;
}

.groups-slider .slider-range {
  width: 100%;
  -webkit-appearance: none;
  appearance: none;
  background: transparent;
  height: 24px;
  cursor: pointer;
  display: block;
}
.groups-slider .slider-range::-webkit-slider-runnable-track {
  height: 4px;
  background: linear-gradient(to right, var(--text-mute), var(--accent));
  border-radius: 2px;
}
.groups-slider .slider-range::-moz-range-track {
  height: 4px;
  background: linear-gradient(to right, var(--text-mute), var(--accent));
  border-radius: 2px;
  border: none;
}
.groups-slider .slider-range::-webkit-slider-thumb {
  -webkit-appearance: none;
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--text-hi);
  border: 2px solid var(--accent);
  margin-top: -7px;
  cursor: grab;
  box-shadow: 0 0 0 3px rgba(107, 209, 255, 0.15);
  transition: transform 0.1s;
}
.groups-slider .slider-range:active::-webkit-slider-thumb {
  cursor: grabbing;
  transform: scale(1.15);
}
.groups-slider .slider-range::-moz-range-thumb {
  width: 18px;
  height: 18px;
  border-radius: 50%;
  background: var(--text-hi);
  border: 2px solid var(--accent);
  cursor: grab;
  box-shadow: 0 0 0 3px rgba(107, 209, 255, 0.15);
}
.groups-slider .slider-ticks {
  display: flex;
  justify-content: space-between;
  margin-top: 4px;
  color: var(--text-mute);
  font-size: 9px;
  font-variant-numeric: tabular-nums;
}

.groups-tabs {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.groups-tab {
  background: var(--bg-surface);
  color: var(--text-base);
  border: 1px solid var(--border);
  padding: 10px 16px;
  font-family: inherit;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s ease;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  gap: 2px;
  min-width: 110px;
}
.groups-tab .t {
  color: var(--text-bright);
  font-weight: 500;
  font-size: 13px;
}
.groups-tab .c {
  color: var(--text-mute);
  font-size: 10px;
}
.groups-tab:hover {
  background: var(--bg-hover);
  border-color: var(--text-mute);
}
.groups-tab.active {
  background: var(--bg-hover);
  border-color: var(--text-hi);
}
.groups-tab.active .t { color: var(--accent); }

.groups-list {
  display: flex;
  flex-direction: column;
  gap: 2px;
  border-top: 1px solid var(--border);
  margin-top: 12px;
}
.group-row {
  border-bottom: 1px solid #141414;
}
.group-head {
  display: flex;
  align-items: center;
  gap: 16px;
  padding: 12px 14px;
  cursor: pointer;
  transition: background 0.1s;
}
.group-head:hover { background: var(--bg-hover); }
.group-row.expanded .group-head { background: var(--bg-hover); }
.group-arrow {
  color: var(--text-mute);
  font-size: 9px;
  width: 12px;
  display: inline-block;
}
.group-label {
  color: var(--text-bright);
  font-size: 13px;
  font-weight: 500;
  flex: 1;
  max-width: 320px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.group-size {
  color: var(--text-mute);
  font-size: 11px;
  min-width: 80px;
}
.group-fit {
  color: var(--text-base);
  font-size: 11px;
  font-variant-numeric: tabular-nums;
  margin-left: auto;
}
.group-fit em {
  color: var(--text-mute);
  font-style: normal;
  margin-right: 2px;
}

.group-body {
  padding: 4px 14px 16px 30px;
  background: #0b0b0b;
}
table.group-members {
  width: 100%;
  border-collapse: collapse;
  font-size: 11px;
}
table.group-members thead th {
  text-align: left;
  color: var(--text-mute);
  font-size: 9px;
  font-weight: 500;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  padding: 8px 10px;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
table.group-members thead th.n { text-align: right; }
table.group-members tbody tr { border-bottom: 1px solid #141414; }
table.group-members tbody tr:hover { background: var(--bg-hover); }
table.group-members tbody td {
  padding: 6px 10px;
  color: var(--text-base);
}
table.group-members tbody td.n {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
table.group-members tbody td.num-pos { color: var(--pos); }
table.group-members tbody td.num-neg { color: var(--neg); }
table.group-members tbody td.fid {
  word-break: break-all;
  max-width: 220px;
}
table.group-members tbody td.fid a {
  color: var(--text-bright);
}
table.group-members tbody td.fid a:hover { color: var(--accent); }
table.group-members tbody td.sub {
  color: var(--text-mute);
  max-width: 160px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
table.group-members tbody td.act {
  text-align: right;
}
table.group-members .mini-btn {
  color: var(--accent);
  border: 1px solid var(--border);
  padding: 2px 6px;
  font-size: 9px;
  cursor: pointer;
}
table.group-members .mini-btn:hover {
  background: var(--bg-hover);
  border-color: var(--accent);
}
"""

MAIN_JS = r"""
(function() {
  const views = {
    graph:   { file: 'views/graph.html',   kind: 'iframe' },
    map:     { file: 'views/map.html',     kind: 'iframe' },
    heatmap: { file: 'views/heatmap.png',  kind: 'image' },
    groups:  { kind: 'groups' },
  };

  const container = document.getElementById('view-container');
  const viewLinks = document.querySelectorAll('.view-list li');
  let currentIframe = null;
  let pendingHighlight = null;

  function attachPanZoom(wrap, img) {
    let scale = 1, tx = 0, ty = 0;
    let dragging = false, sx = 0, sy = 0, stx = 0, sty = 0;
    const MIN = 0.2, MAX = 20;
    function apply() {
      img.style.transform = 'translate(' + tx + 'px,' + ty + 'px) scale(' + scale + ')';
    }
    function reset() { scale = 1; tx = 0; ty = 0; apply(); }
    img.addEventListener('load', reset); reset();
    wrap.addEventListener('wheel', function(e) {
      e.preventDefault();
      const rect = wrap.getBoundingClientRect();
      const cx = e.clientX - rect.left - rect.width / 2;
      const cy = e.clientY - rect.top - rect.height / 2;
      const factor = Math.exp(-e.deltaY * (e.ctrlKey ? 0.015 : 0.0025));
      const next = Math.min(MAX, Math.max(MIN, scale * factor));
      const k = next / scale;
      tx = cx - (cx - tx) * k;
      ty = cy - (cy - ty) * k;
      scale = next;
      apply();
    }, { passive: false });
    wrap.addEventListener('mousedown', function(e) {
      dragging = true; sx = e.clientX; sy = e.clientY; stx = tx; sty = ty;
      wrap.classList.add('grabbing');
      e.preventDefault();
    });
    window.addEventListener('mousemove', function(e) {
      if (!dragging) return;
      tx = stx + (e.clientX - sx); ty = sty + (e.clientY - sy); apply();
    });
    window.addEventListener('mouseup', function() {
      dragging = false; wrap.classList.remove('grabbing');
    });
    wrap.addEventListener('dblclick', reset);
    // touch
    let pd = 0, ps = 1, pcx = 0, pcy = 0;
    wrap.addEventListener('touchstart', function(e) {
      if (e.touches.length === 2) {
        const a = e.touches[0], b = e.touches[1];
        pd = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
        ps = scale;
        const r = wrap.getBoundingClientRect();
        pcx = (a.clientX + b.clientX) / 2 - r.left - r.width / 2;
        pcy = (a.clientY + b.clientY) / 2 - r.top - r.height / 2;
      } else if (e.touches.length === 1) {
        dragging = true; sx = e.touches[0].clientX; sy = e.touches[0].clientY;
        stx = tx; sty = ty;
      }
    }, { passive: false });
    wrap.addEventListener('touchmove', function(e) {
      if (e.touches.length === 2) {
        e.preventDefault();
        const a = e.touches[0], b = e.touches[1];
        const d = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
        const next = Math.min(MAX, Math.max(MIN, ps * (d / pd)));
        const k = next / scale;
        tx = pcx - (pcx - tx) * k;
        ty = pcy - (pcy - ty) * k;
        scale = next;
        apply();
      } else if (e.touches.length === 1 && dragging) {
        e.preventDefault();
        tx = stx + (e.touches[0].clientX - sx);
        ty = sty + (e.touches[0].clientY - sy);
        apply();
      }
    }, { passive: false });
    wrap.addEventListener('touchend', function() { dragging = false; });
  }

  function showView(key) {
    const v = views[key];
    if (!v) return;
    viewLinks.forEach(function(el) {
      el.classList.toggle('active', el.dataset.view === key);
    });
    container.innerHTML = '';
    currentIframe = null;
    if (v.kind === 'iframe') {
      const ifr = document.createElement('iframe');
      ifr.className = 'view-iframe';
      ifr.src = v.file;
      ifr.addEventListener('load', function() {
        if (pendingHighlight) {
          try { ifr.contentWindow.postMessage({ type: 'highlight', field_id: pendingHighlight }, '*'); } catch (e) {}
          pendingHighlight = null;
        }
      });
      container.appendChild(ifr);
      currentIframe = ifr;
    } else if (v.kind === 'image') {
      const wrap = document.createElement('div');
      wrap.className = 'view-image-wrap pan-zoom';
      const img = document.createElement('img');
      img.className = 'view-image';
      img.src = v.file;
      img.alt = key;
      img.draggable = false;
      wrap.appendChild(img);
      container.appendChild(wrap);
      attachPanZoom(wrap, img);
    } else if (v.kind === 'groups') {
      renderGroupsPanel(container);
    }
    history.replaceState({}, '', '#' + key);
  }

  // ============ GROUPS panel ============
  let groupsData = null;
  let groupsThreshold = '0.35';
  let groupsExpanded = null;

  function csvEscape(v) {
    if (v === null || v === undefined) return '';
    const s = String(v);
    if (/[",\n\r]/.test(s)) {
      return '"' + s.replace(/"/g, '""') + '"';
    }
    return s;
  }

  function exportGroupsCSV(threshold, groups, topN) {
    const cols = [
      'threshold', 'group_id', 'group_label', 'group_size', 'rank',
      'field_id', 'fitness', 'sharpe', 'turnover', 'returns', 'drawdown',
      'alpha_count', 'category', 'subcategory', 'dataset', 'alpha_id', 'expr'
    ];
    const lines = [cols.join(',')];
    groups.forEach(function(g) {
      const topMembers = (g.members || []).slice(0, topN);
      topMembers.forEach(function(m, i) {
        const row = [
          threshold, g.id, g.label, g.size, (i + 1),
          m.field_id,
          (m.fitness || 0).toFixed(4),
          (m.sharpe || 0).toFixed(4),
          (m.turnover || 0).toFixed(5),
          (m.returns || 0).toFixed(5),
          (m.drawdown || 0).toFixed(4),
          m.alpha_count || 0,
          m.category || '',
          m.subcategory || '',
          m.dataset || '',
          m.alpha_id || '',
          m.expr || ''
        ].map(csvEscape);
        lines.push(row.join(','));
      });
    });
    // UTF-8 BOM → Excel 한글 깨짐 방지
    const blob = new Blob(['\ufeff' + lines.join('\n')], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'field_groups_thr' + threshold + '_top' + topN + '.csv';
    document.body.appendChild(a);
    a.click();
    setTimeout(function() {
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }, 100);
  }

  function renderGroupsPanel(container) {
    const panel = document.createElement('div');
    panel.className = 'groups-panel';
    panel.innerHTML = '<div class="groups-loading">loading groups...</div>';
    container.appendChild(panel);

    const render = function() {
      panel.innerHTML = '';
      if (!groupsData) {
        panel.innerHTML = '<div class="groups-loading">no data</div>';
        return;
      }

      // header
      const header = document.createElement('div');
      header.className = 'groups-header';
      const title = document.createElement('div');
      title.className = 'groups-title';
      title.textContent = 'GROUPS';
      header.appendChild(title);

      const explain = document.createElement('div');
      explain.className = 'groups-explain';
      explain.textContent = 'pick minimum combined correlation — lower = fewer / larger groups, higher = more / tighter groups';
      header.appendChild(explain);

      // threshold slider
      const thresholds = Object.keys(groupsData).sort(function(a, b) {
        return parseFloat(a) - parseFloat(b);
      });
      if (thresholds.indexOf(groupsThreshold) === -1) {
        // 기본값이 없으면 중간값 근처로
        groupsThreshold = thresholds[Math.floor(thresholds.length / 2)] || thresholds[0];
      }
      const currentIdx = thresholds.indexOf(groupsThreshold);
      const currentCount = (groupsData[groupsThreshold] || []).length;

      const sliderWrap = document.createElement('div');
      sliderWrap.className = 'groups-slider';
      sliderWrap.innerHTML =
        '<div class="slider-top">' +
          '<span class="slider-label">min combined ≥</span>' +
          '<span class="slider-value" id="grp-thr-val">' + groupsThreshold + '</span>' +
          '<span class="slider-count"><b id="grp-count">' + currentCount + '</b> groups</span>' +
        '</div>' +
        '<input type="range" class="slider-range" id="grp-slider" ' +
          'min="0" max="' + (thresholds.length - 1) + '" step="1" value="' + currentIdx + '">' +
        '<div class="slider-ticks">' +
          '<span>' + thresholds[0] + '</span>' +
          '<span>' + thresholds[Math.floor(thresholds.length / 2)] + '</span>' +
          '<span>' + thresholds[thresholds.length - 1] + '</span>' +
        '</div>';
      header.appendChild(sliderWrap);

      // export CSV
      const exportBar = document.createElement('div');
      exportBar.className = 'groups-export';
      exportBar.innerHTML =
        '<span class="ex-label">export top</span>' +
        '<input type="number" id="grp-topn" class="ex-input" value="1" min="1" max="50" step="1">' +
        '<span class="ex-label">/ group (current threshold)</span>' +
        '<button class="ex-btn" id="grp-export">⬇ download CSV</button>';
      header.appendChild(exportBar);

      panel.appendChild(header);

      // slider 이벤트 — live update 없이 group list 만 다시 그림
      const sliderEl = sliderWrap.querySelector('#grp-slider');
      const thrValEl = sliderWrap.querySelector('#grp-thr-val');
      const countEl = sliderWrap.querySelector('#grp-count');
      sliderEl.addEventListener('input', function() {
        const idx = parseInt(sliderEl.value, 10);
        groupsThreshold = thresholds[idx];
        groupsExpanded = null;
        thrValEl.textContent = groupsThreshold;
        countEl.textContent = (groupsData[groupsThreshold] || []).length;
        renderList();
      });

      // group list 컨테이너
      const list = document.createElement('div');
      list.className = 'groups-list';
      panel.appendChild(list);

      function renderList() {
        list.innerHTML = '';
        const groups = groupsData[groupsThreshold] || [];
        groups.forEach(function(g) {
        const row = document.createElement('div');
        row.className = 'group-row' + (groupsExpanded === g.id ? ' expanded' : '');

        const head = document.createElement('div');
        head.className = 'group-head';
        head.addEventListener('click', function() {
          groupsExpanded = (groupsExpanded === g.id) ? null : g.id;
          renderList();
        });

        const arrow = document.createElement('span');
        arrow.className = 'group-arrow';
        arrow.textContent = groupsExpanded === g.id ? '▼' : '▶';

        const label = document.createElement('span');
        label.className = 'group-label';
        label.textContent = g.label || '—';

        const size = document.createElement('span');
        size.className = 'group-size';
        size.textContent = g.size + ' fields';

        const fit = document.createElement('span');
        fit.className = 'group-fit';
        fit.innerHTML = '<em>max fit </em>' + g.max_fitness.toFixed(2) +
                        '  <em>avg </em>' + g.avg_fitness.toFixed(2);

        head.appendChild(arrow);
        head.appendChild(label);
        head.appendChild(size);
        head.appendChild(fit);
        row.appendChild(head);

        if (groupsExpanded === g.id) {
          const body = document.createElement('div');
          body.className = 'group-body';

          // mini table (이미 fitness desc 로 정렬되어 있음)
          const tbl = document.createElement('table');
          tbl.className = 'group-members';
          tbl.innerHTML = '<thead><tr>' +
            '<th>field</th>' +
            '<th class="n">fitness</th>' +
            '<th class="n">sharpe</th>' +
            '<th class="n">turnover</th>' +
            '<th class="n">α</th>' +
            '<th>category</th>' +
            '<th>subcategory</th>' +
            '<th></th>' +
            '</tr></thead>';
          const tb = document.createElement('tbody');
          g.members.forEach(function(m) {
            const tr = document.createElement('tr');

            const tdF = document.createElement('td');
            tdF.className = 'fid';
            const a = document.createElement('a');
            a.href = 'detail.html?fid=' + encodeURIComponent(m.field_id);
            a.target = '_blank';
            a.textContent = m.field_id;
            tdF.appendChild(a);
            tr.appendChild(tdF);

            function numCell(v, dec, goodGt) {
              const td = document.createElement('td');
              td.className = 'n';
              td.textContent = (v || 0).toFixed(dec);
              if (goodGt !== undefined && v >= goodGt) td.classList.add('num-pos');
              return td;
            }
            tr.appendChild(numCell(m.fitness, 2, 1.0));
            tr.appendChild(numCell(m.sharpe, 2, 1.25));
            tr.appendChild(numCell(m.turnover, 3));

            const tdA = document.createElement('td');
            tdA.className = 'n';
            tdA.textContent = m.alpha_count || 0;
            tr.appendChild(tdA);

            const tdC = document.createElement('td');
            tdC.className = 'sub';
            tdC.textContent = m.category || '—';
            tr.appendChild(tdC);

            const tdS = document.createElement('td');
            tdS.className = 'sub';
            tdS.textContent = m.subcategory || '—';
            tr.appendChild(tdS);

            const tdAct = document.createElement('td');
            tdAct.className = 'act';
            const act = document.createElement('a');
            act.className = 'mini-btn';
            act.textContent = 'focus';
            act.addEventListener('click', function(ev) {
              ev.preventDefault();
              highlightOnCurrent(m.field_id);
            });
            tdAct.appendChild(act);
            tr.appendChild(tdAct);

            tb.appendChild(tr);
          });
          tbl.appendChild(tb);
          body.appendChild(tbl);
          row.appendChild(body);
        }

          list.appendChild(row);
        });
      }  // end renderList
      renderList();

      // export CSV handler
      const exportBtn = panel.querySelector('#grp-export');
      const topnInput = panel.querySelector('#grp-topn');
      if (exportBtn) {
        exportBtn.addEventListener('click', function() {
          const n = Math.max(1, parseInt(topnInput.value || '1', 10));
          exportGroupsCSV(groupsThreshold, groupsData[groupsThreshold] || [], n);
        });
      }
    };

    if (groupsData) {
      render();
    } else {
      fetch('assets/groups.json')
        .then(function(r) { return r.json(); })
        .then(function(d) {
          groupsData = d;
          if (groupsData && !groupsData[groupsThreshold]) {
            groupsThreshold = Object.keys(groupsData).sort()[0];
          }
          render();
        })
        .catch(function(e) {
          panel.innerHTML = '<div class="groups-loading">failed: ' + e.message + '</div>';
        });
    }
  }

  function currentKey() {
    return (location.hash || '#graph').slice(1);
  }

  function highlightOnCurrent(fid) {
    const key = currentKey();
    if (views[key] && views[key].kind === 'iframe' && currentIframe) {
      try {
        currentIframe.contentWindow.postMessage({ type: 'highlight', field_id: fid }, '*');
      } catch (e) {}
    } else {
      // heatmap 활성 상태 → graph 로 스위칭 후 반영
      pendingHighlight = fid;
      showView('graph');
    }
  }

  viewLinks.forEach(function(el) {
    el.addEventListener('click', function() { showView(el.dataset.view); });
  });

  const initial = currentKey();
  showView(views[initial] ? initial : 'graph');

  fetch('assets/data.json')
    .then(function(r) { return r.json(); })
    .then(function(data) {
      window.FIELD_DATA = data;
      const input = document.getElementById('search-input');
      const results = document.getElementById('search-results');

      function render(matches) {
        results.innerHTML = '';
        if (!matches.length) {
          const e = document.createElement('div');
          e.className = 'search-empty';
          e.textContent = 'no match';
          results.appendChild(e);
          return;
        }
        matches.forEach(function(m) {
          const div = document.createElement('div');
          div.className = 'search-result';
          div.dataset.fid = m.field_id;

          const idLine = document.createElement('div');
          idLine.className = 'sr-id';
          idLine.textContent = m.field_id;
          div.appendChild(idLine);

          const meta = document.createElement('div');
          meta.className = 'sr-meta';
          const cat = document.createElement('span');
          cat.className = 'cat';
          cat.textContent = (m.category || '—') + (m.subcategory ? ' / ' + m.subcategory : '');
          const num = document.createElement('span');
          num.className = 'num';
          const parts = [];
          if (m.alpha_count) parts.push(m.alpha_count + 'α');
          num.textContent = parts.join(' · ');
          meta.appendChild(cat); meta.appendChild(num);
          div.appendChild(meta);

          const stats = document.createElement('div');
          stats.className = 'sr-stats';
          function stat(label, val, cls) {
            const s = document.createElement('span');
            s.className = 'sr-stat' + (cls ? ' ' + cls : '');
            s.innerHTML = '<em>' + label + '</em>' + val;
            return s;
          }
          const sh = m.sharpe !== undefined ? Number(m.sharpe).toFixed(2) : '—';
          const fi = m.fitness !== undefined ? Number(m.fitness).toFixed(2) : '—';
          const to = m.turnover !== undefined ? Number(m.turnover).toFixed(3) : '—';
          stats.appendChild(stat('sh ', sh, Number(m.sharpe) >= 1.25 ? 'good' : ''));
          stats.appendChild(stat('fit ', fi, Number(m.fitness) >= 1.0 ? 'good' : ''));
          stats.appendChild(stat('to ', to, ''));
          div.appendChild(stats);

          const actions = document.createElement('div');
          actions.className = 'sr-actions';
          const focusBtn = document.createElement('a');
          focusBtn.textContent = 'focus';
          focusBtn.addEventListener('click', function(ev) {
            ev.stopPropagation();
            highlightOnCurrent(m.field_id);
          });
          const detailBtn = document.createElement('a');
          detailBtn.textContent = '자세히 보기';
          detailBtn.href = 'detail.html?fid=' + encodeURIComponent(m.field_id);
          detailBtn.target = '_blank';
          detailBtn.rel = 'noopener';
          const copyBtn = document.createElement('a');
          copyBtn.textContent = 'copy';
          copyBtn.addEventListener('click', function(ev) {
            ev.stopPropagation();
            navigator.clipboard.writeText(m.field_id);
            copyBtn.textContent = '✓';
            setTimeout(function() { copyBtn.textContent = 'copy'; }, 1000);
          });
          actions.appendChild(focusBtn);
          actions.appendChild(detailBtn);
          actions.appendChild(copyBtn);
          div.appendChild(actions);

          div.addEventListener('click', function() { highlightOnCurrent(m.field_id); });
          results.appendChild(div);
        });
      }

      function score(f, q) {
        const fid = (f.field_id || '').toLowerCase();
        const sub = (f.subcategory || '').toLowerCase();
        const dset = (f.dataset || '').toLowerCase();
        const cat = (f.category || '').toLowerCase();
        if (fid === q) return 100;
        if (fid.startsWith(q)) return 80;
        if (fid.includes(q)) return 60;
        if (sub.includes(q)) return 40;
        if (dset.includes(q)) return 30;
        if (cat.includes(q)) return 20;
        return 0;
      }

      input.addEventListener('input', function() {
        const q = input.value.trim().toLowerCase();
        if (q.length < 2) { results.innerHTML = ''; return; }
        const scored = (data.fields || [])
          .map(function(f) { return { f: f, s: score(f, q) }; })
          .filter(function(x) { return x.s > 0; })
          .sort(function(a, b) { return b.s - a.s || b.f.alpha_count - a.f.alpha_count; })
          .slice(0, 30)
          .map(function(x) { return x.f; });
        render(scored);
      });
    })
    .catch(function() {});
})();
"""

DETAIL_JS = r"""
(function() {
  const params = new URLSearchParams(location.search);
  const fid = params.get('fid');
  const titleEl = document.getElementById('detail-title');
  const metaEl = document.getElementById('detail-meta');
  const bodyEl = document.getElementById('detail-table-body');
  const headEl = document.getElementById('detail-table-head');
  const loadingEl = document.getElementById('detail-loading');
  const errorEl = document.getElementById('detail-error');
  const tableEl = document.getElementById('detail-table');
  const filterEl = document.getElementById('detail-filter');
  const countEl = document.getElementById('detail-count');

  if (!fid) {
    errorEl.textContent = 'no field_id in URL. use detail.html?fid=...';
    errorEl.style.display = 'block';
    loadingEl.style.display = 'none';
    return;
  }
  titleEl.textContent = fid;
  document.title = fid + ' — Field Detail';

  // 컬럼 정의: [key, label, formatter, class]
  const COLS = [
    { key: 'field_id', label: 'field',      fmt: function(v, row) {
        const a = document.createElement('a');
        a.href = 'detail.html?fid=' + encodeURIComponent(v);
        a.textContent = v;
        return a;
      }, cls: 'fid' },
    { key: 'combined', label: 'combined',   fmt: function(v) { return v.toFixed(4); }, num: true },
    { key: 'pnl',      label: 'pnl corr',   fmt: function(v) { return v.toFixed(4); }, num: true },
    { key: 'text',     label: 'text sim',   fmt: function(v) { return v.toFixed(4); }, num: true },
    { key: 'fitness',  label: 'fitness',    fmt: function(v) { return (v || 0).toFixed(2); }, num: true },
    { key: 'sharpe',   label: 'sharpe',     fmt: function(v) { return (v || 0).toFixed(2); }, num: true },
    { key: 'turnover', label: 'turnover',   fmt: function(v) { return (v || 0).toFixed(3); }, num: true },
    { key: 'category', label: 'category',   fmt: function(v) { return v || '—'; }, cls: 'cat' },
    { key: 'subcategory', label: 'subcategory', fmt: function(v) { return v || '—'; }, cls: 'cat' },
    { key: 'alpha_count', label: 'α count', fmt: function(v) { return String(v || 0); }, num: true },
  ];

  let rows = [];
  let sortKey = 'combined';
  let sortDir = -1;  // -1 desc, 1 asc
  let filterQ = '';

  function renderHead() {
    headEl.innerHTML = '';
    const tr = document.createElement('tr');
    COLS.forEach(function(c) {
      const th = document.createElement('th');
      th.textContent = c.label;
      if (c.key === sortKey) {
        th.classList.add('sorted');
        const arrow = document.createElement('span');
        arrow.className = 'arrow';
        arrow.textContent = sortDir < 0 ? '▼' : '▲';
        th.appendChild(arrow);
      }
      th.addEventListener('click', function() {
        if (sortKey === c.key) sortDir *= -1;
        else { sortKey = c.key; sortDir = c.num ? -1 : 1; }
        renderHead();
        renderBody();
      });
      tr.appendChild(th);
    });
    headEl.appendChild(tr);
  }

  function renderBody() {
    const filtered = filterQ
      ? rows.filter(function(r) {
          return (r.field_id || '').toLowerCase().includes(filterQ) ||
                 (r.category || '').toLowerCase().includes(filterQ) ||
                 (r.subcategory || '').toLowerCase().includes(filterQ);
        })
      : rows;

    const sorted = filtered.slice().sort(function(a, b) {
      const va = a[sortKey], vb = b[sortKey];
      if (va === undefined || va === null) return 1;
      if (vb === undefined || vb === null) return -1;
      if (typeof va === 'number' && typeof vb === 'number') return (va - vb) * sortDir;
      return String(va).localeCompare(String(vb)) * sortDir;
    });

    countEl.textContent = sorted.length + ' / ' + rows.length + ' rows';
    bodyEl.innerHTML = '';
    const frag = document.createDocumentFragment();
    sorted.forEach(function(r) {
      const tr = document.createElement('tr');
      COLS.forEach(function(c) {
        const td = document.createElement('td');
        const v = r[c.key];
        const out = c.fmt(v, r);
        if (typeof out === 'string') td.textContent = out;
        else td.appendChild(out);
        if (c.cls) td.classList.add(c.cls);
        if (c.num && typeof v === 'number') {
          if (v > 0.001) td.classList.add('num-pos');
          else if (v < -0.001) td.classList.add('num-neg');
        }
        tr.appendChild(td);
      });
      frag.appendChild(tr);
    });
    bodyEl.appendChild(frag);
  }

  filterEl.addEventListener('input', function() {
    filterQ = filterEl.value.trim().toLowerCase();
    renderBody();
  });

  // fetch neighbor data
  fetch('assets/neighbors/' + encodeURIComponent(fid) + '.json')
    .then(function(r) {
      if (!r.ok) throw new Error('HTTP ' + r.status);
      return r.json();
    })
    .then(function(data) {
      loadingEl.style.display = 'none';
      tableEl.style.display = '';
      const self = data.self || {};
      const metaParts = [
        ['category', self.category || '—'],
        ['subcategory', self.subcategory || '—'],
        ['dataset', self.dataset || '—'],
        ['fitness', self.fitness !== undefined ? Number(self.fitness).toFixed(3) : '—'],
        ['sharpe', self.sharpe !== undefined ? Number(self.sharpe).toFixed(3) : '—'],
        ['turnover', self.turnover !== undefined ? Number(self.turnover).toFixed(4) : '—'],
        ['α count', self.alpha_count || 0],
        ['neighbors', (data.rows || []).length],
      ];
      metaEl.innerHTML = '';
      metaParts.forEach(function(p) {
        const span = document.createElement('span');
        const l = document.createElement('span');
        l.className = 'label'; l.textContent = p[0] + ':';
        const v = document.createElement('span');
        v.className = 'val'; v.textContent = p[1];
        span.appendChild(l); span.appendChild(v);
        metaEl.appendChild(span);
      });

      rows = (data.rows || []).map(function(row) {
        return {
          field_id: row[0],
          combined: row[1],
          pnl: row[2],
          text: row[3],
          sharpe: row[4],
          fitness: row[5],
          turnover: row[6],
          category: row[7],
          subcategory: row[8],
          alpha_count: row[9],
        };
      });
      renderHead();
      renderBody();
    })
    .catch(function(e) {
      loadingEl.style.display = 'none';
      errorEl.textContent = 'failed to load neighbors: ' + e.message;
      errorEl.style.display = 'block';
    });
})();
"""


GROUP_THRESHOLDS = [round(0.05 * i, 2) for i in range(1, 20)]  # 0.05 ~ 0.95, 0.05 단위


def _label_group(members: list[dict]) -> str:
    """그룹의 한 단어 라벨 — 지배적 subcategory → category → 'mixed'."""
    from collections import Counter
    n = len(members)
    if n == 0:
        return "empty"
    subs = Counter(m.get("subcategory") or "" for m in members)
    sub_top, sub_cnt = subs.most_common(1)[0]
    if sub_top and sub_cnt / n >= 0.6:
        # 짧게 줄이기 (첫 2단어)
        words = sub_top.split()
        return " ".join(words[:2]) if words else sub_top

    cats = Counter(m.get("category") or "" for m in members)
    cat_top, cat_cnt = cats.most_common(1)[0]
    if cat_top and cat_cnt / n >= 0.6:
        return cat_top

    # 혼합: top2 카테고리 조합
    top2 = [c[0] for c in cats.most_common(2) if c[0]]
    if len(top2) >= 2:
        return " / ".join(top2)
    if top2:
        return top2[0] + " mixed"
    return "mixed"


def build_groups_json(
    assets_dir: Path,
    data_dir: Path,
    field_index: list[dict],
    sim_meta_by_fid: dict,
) -> dict:
    """Agglomerative (complete linkage) 클러스터링 × 4 threshold."""
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform

    combined_path = data_dir / "similarity_combined.npy"
    ids_path = data_dir / "similarity_field_ids.json"
    if not combined_path.exists() or not ids_path.exists():
        log_line("09d", "similarity matrices 없음 — groups skip")
        return {}

    combined = np.load(combined_path).astype(np.float32)
    sim_ids: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))
    field_meta = {f["field_id"]: f for f in field_index}

    # 거리 행렬
    d = 1.0 - combined
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    d = np.clip(d, 0.0, None)
    condensed = squareform(d, checks=False)
    Z = linkage(condensed, method="complete")

    result: dict[str, list[dict]] = {}
    for thr in GROUP_THRESHOLDS:
        cutoff = 1.0 - thr  # combined ≥ thr  ⟺  distance ≤ 1-thr
        labels = fcluster(Z, t=cutoff, criterion="distance")
        groups: dict[int, list[dict]] = {}
        for i, lbl in enumerate(labels):
            fid = sim_ids[i]
            m = field_meta.get(fid, {})
            sm = sim_meta_by_fid.get(fid, {})
            groups.setdefault(int(lbl), []).append({
                "field_id": fid,
                "category": m.get("category", ""),
                "subcategory": m.get("subcategory", ""),
                "dataset": m.get("dataset", ""),
                "fitness": float(m.get("fitness", 0.0) or 0.0),
                "sharpe": float(m.get("sharpe", 0.0) or 0.0),
                "turnover": float(m.get("turnover", 0.0) or 0.0),
                "returns": float(sm.get("returns", 0.0) or 0.0),
                "drawdown": float(sm.get("drawdown", 0.0) or 0.0),
                "alpha_count": int(m.get("alpha_count", 0) or 0),
                "expr": sm.get("expr", ""),
                "alpha_id": sm.get("alpha_id", ""),
            })

        # 그룹 정렬: 크기 내림차순
        group_list = []
        for gid, members in groups.items():
            members.sort(key=lambda x: (-x["fitness"], -x["sharpe"]))
            group_list.append({
                "id": gid,
                "label": _label_group(members),
                "size": len(members),
                "avg_fitness": round(
                    sum(m["fitness"] for m in members) / max(1, len(members)),
                    3,
                ),
                "max_fitness": round(max((m["fitness"] for m in members), default=0.0), 3),
                "top_category": members[0]["category"] if members else "",
                "members": members,
            })
        group_list.sort(key=lambda g: (-g["size"], -g["max_fitness"]))
        result[f"{thr:.2f}"] = group_list

    out_path = assets_dir / "groups.json"
    out_path.write_text(
        json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    summary = {t: f"{len(g)} groups" for t, g in result.items()}
    log_line("09d", f"groups built: {summary}")
    return result


def build_neighbor_files(
    assets_dir: Path,
    data_dir: Path,
    field_index: list[dict],
    sim_meta: dict,
    top_k: int = 500,
) -> int:
    """각 필드의 이웃 상관 리스트를 JSON 파일로 dump."""
    combined_path = data_dir / "similarity_combined.npy"
    pnl_path = data_dir / "similarity_pnl.npy"
    text_path = data_dir / "similarity_text.npy"
    ids_path = data_dir / "similarity_field_ids.json"

    if not combined_path.exists() or not ids_path.exists():
        log_line("09d", "similarity matrices 없음 — neighbors skip")
        return 0

    combined = np.load(combined_path).astype(np.float32)
    pnl = np.load(pnl_path).astype(np.float32) if pnl_path.exists() else combined
    text = np.load(text_path).astype(np.float32) if text_path.exists() else combined
    sim_ids: list[str] = json.loads(ids_path.read_text(encoding="utf-8"))
    id_to_idx = {fid: i for i, fid in enumerate(sim_ids)}

    field_meta = {f["field_id"]: f for f in field_index}

    neighbors_dir = assets_dir / "neighbors"
    # 기존 파일 정리 (stale neighbors 제거)
    if neighbors_dir.exists():
        for p in neighbors_dir.glob("*.json"):
            p.unlink()
    neighbors_dir.mkdir(parents=True, exist_ok=True)

    n_written = 0
    for fid in sim_ids:
        i = id_to_idx[fid]
        # 자기 자신 제외하고 combined 내림차순 정렬
        scores = combined[i].copy()
        scores[i] = -np.inf
        order = np.argsort(-scores)[: top_k]

        self_meta = field_meta.get(fid, {})
        self_sm = sim_meta.get(fid, {})

        rows = []
        for j in order:
            if scores[j] == -np.inf:
                continue
            oid = sim_ids[int(j)]
            om = field_meta.get(oid, {})
            osm = sim_meta.get(oid, {})
            rows.append([
                oid,
                float(combined[i, j]),
                float(pnl[i, j]),
                float(text[i, j]),
                float(osm.get("sharpe", 0.0)),
                float(osm.get("fitness", 0.0)),
                float(osm.get("turnover", 0.0)),
                om.get("category", ""),
                om.get("subcategory", ""),
                int(om.get("alpha_count", 0)),
            ])

        payload = {
            "self": {
                "field_id": fid,
                "category": self_meta.get("category", ""),
                "subcategory": self_meta.get("subcategory", ""),
                "dataset": self_meta.get("dataset", ""),
                "alpha_count": int(self_meta.get("alpha_count", 0)),
                "sharpe": float(self_sm.get("sharpe", 0.0)),
                "fitness": float(self_sm.get("fitness", 0.0)),
                "turnover": float(self_sm.get("turnover", 0.0)),
                "returns": float(self_sm.get("returns", 0.0)),
                "status": self_sm.get("status", ""),
                "alpha_id": self_sm.get("alpha_id", ""),
            },
            "rows": rows,
        }
        (neighbors_dir / f"{fid}.json").write_text(
            json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        n_written += 1
    return n_written


def main():
    config = load_config()
    log_line("09d", "start")

    site_dir = resolve_path("output/site")
    assets_dir = site_dir / "assets"
    views_dir = site_dir / "views"
    assets_dir.mkdir(parents=True, exist_ok=True)
    views_dir.mkdir(parents=True, exist_ok=True)

    data_dir = resolve_path(config["paths"]["data_dir"])
    corpus_path = resolve_path(config["paths"]["corpus_file"])
    sim_meta_path = data_dir / "similarity_meta.json"
    clusters_path = data_dir / "clusters_behavior.json"
    field_meta_path = data_dir / "single_field_meta.jsonl"

    # CSS / JS
    (assets_dir / "style.css").write_text(STYLE_CSS, encoding="utf-8")
    (assets_dir / "main.js").write_text(MAIN_JS, encoding="utf-8")
    (assets_dir / "detail.js").write_text(DETAIL_JS, encoding="utf-8")

    # single_field_meta 로드
    sim_meta_by_fid: dict[str, dict] = {}
    n_sim_done = 0
    if field_meta_path.exists():
        with open(field_meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    sim_meta_by_fid[r["field_id"]] = r
                    n_sim_done += 1

    # field_index — 검색/카테고리용
    field_index: list[dict] = []
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    sm = sim_meta_by_fid.get(r["field_id"], {})
                    field_index.append({
                        "field_id": r["field_id"],
                        "category": r.get("category_name", ""),
                        "subcategory": r.get("subcategory_name", ""),
                        "dataset": r.get("dataset_name", ""),
                        "alpha_count": r.get("alpha_count", 0),
                        "sharpe": round(float(sm.get("sharpe", 0.0)), 3),
                        "fitness": round(float(sm.get("fitness", 0.0)), 3),
                        "turnover": round(float(sm.get("turnover", 0.0)), 4),
                        "status": sm.get("status", ""),
                    })

    sim_summary = {}
    if sim_meta_path.exists():
        sim_summary = json.loads(sim_meta_path.read_text(encoding="utf-8"))

    clusters_info = {}
    if clusters_path.exists():
        cb = json.loads(clusters_path.read_text(encoding="utf-8"))
        clusters_info = {
            "n_clusters": cb.get("n_clusters", 0),
            "n_noise": cb.get("n_noise", 0),
        }

    cat_counter = Counter(f["category"] for f in field_index)
    categories = [{"name": k, "count": v} for k, v in cat_counter.most_common()]

    data_json = {
        "n_fields_total": len(field_index),
        "n_fields_simulated": n_sim_done,
        "similarity": sim_summary,
        "clusters": clusters_info,
        "categories": categories,
        "fields": field_index,
    }
    (assets_dir / "data.json").write_text(
        json.dumps(data_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # 이웃 파일 (per-field)
    log_line("09d", "building neighbor files...")
    n_neighbors = build_neighbor_files(
        assets_dir, data_dir, field_index, sim_meta_by_fid, top_k=500,
    )
    log_line("09d", f"neighbors written: {n_neighbors}")

    # 그룹 (threshold별)
    log_line("09d", "building groups.json...")
    build_groups_json(assets_dir, data_dir, field_index, sim_meta_by_fid)

    # index.html
    cat_items = "\n".join(
        f'<li><span>{c["name"] or "—"}</span><span class="count">{c["count"]}</span></li>'
        for c in categories
    )
    index_html = f"""<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Field Graph</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="app">
  <header class="header">
    <div class="brand">FIELD GRAPH</div>
    <div class="meta">
      fields {data_json['n_fields_total']}
      · simulated {data_json['n_fields_simulated']}
      · clusters {clusters_info.get('n_clusters', '—')}
    </div>
  </header>

  <aside class="sidebar">
    <div class="section">
      <div class="section-title">OVERVIEW</div>
      <div class="stat-row"><span>fields</span><span class="num">{data_json['n_fields_total']}</span></div>
      <div class="stat-row"><span>simulated</span><span class="num">{data_json['n_fields_simulated']}</span></div>
      <div class="stat-row"><span>clusters</span><span class="num">{clusters_info.get('n_clusters', '—')}</span></div>
      <div class="stat-row"><span>noise</span><span class="num">{clusters_info.get('n_noise', '—')}</span></div>
    </div>

    <div class="section">
      <div class="section-title">VIEW</div>
      <ul class="view-list">
        <li data-view="graph"><span class="view-id">01</span>GRAPH</li>
        <li data-view="map"><span class="view-id">02</span>MAP</li>
        <li data-view="heatmap"><span class="view-id">03</span>HEATMAP</li>
        <li data-view="groups"><span class="view-id">04</span>GROUPS</li>
      </ul>
    </div>

    <div class="section">
      <div class="section-title">SEARCH</div>
      <input id="search-input" class="search-input" placeholder="field / dataset / subcategory..." autocomplete="off">
      <div id="search-results" class="search-results"></div>
    </div>

    <div class="section">
      <div class="section-title">CATEGORIES</div>
      <ul class="category-list">
        {cat_items}
      </ul>
    </div>
  </aside>

  <main class="main">
    <div id="view-container" class="view-container">
      <div class="empty-state"><div class="dot"></div><span>loading...</span></div>
    </div>
  </main>

  <footer class="footer">
    generated · pnl_weight={sim_summary.get('pnl_weight', '—')}
    · text_weight={sim_summary.get('text_weight', '—')}
    · demean={sim_summary.get('pnl_demean', '—')}
    · HDBSCAN
  </footer>
</div>
<script src="assets/main.js"></script>
</body>
</html>
"""
    (site_dir / "index.html").write_text(index_html, encoding="utf-8")

    # detail.html
    detail_html = """<!doctype html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>Field Detail</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
<link rel="stylesheet" href="assets/style.css">
</head>
<body>
<div class="detail-app">
  <div class="detail-header">
    <a class="detail-back" href="index.html">← back to graph</a>
    <h1 id="detail-title" class="detail-title">...</h1>
    <div id="detail-meta" class="detail-meta"></div>
  </div>

  <div class="detail-controls">
    <input id="detail-filter" class="filter-input" placeholder="filter by field / category / subcategory..." autocomplete="off">
    <span id="detail-count" class="count-note">loading...</span>
  </div>

  <div id="detail-loading" class="detail-loading">loading neighbors...</div>
  <div id="detail-error" class="detail-error" style="display:none"></div>

  <table id="detail-table" class="neighbors" style="display:none">
    <thead id="detail-table-head"></thead>
    <tbody id="detail-table-body"></tbody>
  </table>
</div>
<script src="assets/detail.js"></script>
</body>
</html>
"""
    (site_dir / "detail.html").write_text(detail_html, encoding="utf-8")

    log_line("09d", f"site assembled → {site_dir}")
    update_step(config, STEP, status="done", progress_value=1.0,
                message=f"site ready at {site_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
