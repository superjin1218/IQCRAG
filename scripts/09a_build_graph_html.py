"""
Step 09a — PyVis 인터랙티브 그래프 HTML (모노크롬).

특징:
  - 배경 완전 블랙
  - 노드: 빈 원 + 테두리 (stroke 명도로 중요도/클러스터 표현)
  - 엣지: 유사도 → 명도로 (0.0=#2a2a2a, 1.0=#ffffff)
  - 라벨: 센트로이드/주요 노드만 보임, 나머지는 hover 시만
  - 계층형 펼치기 모드: 센트로이드들만 먼저 보여주고, 클릭 시 그 클러스터 멤버 펼침
    (단순히 서브셋 표시 방식)

출력:
  output/site/views/graph.html
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, resolve_path, ensure_parent, log_line, update_step,
)

STEP = "09_visualize"


def _grey(value: float, lo: int = 42, hi: int = 255) -> str:
    """value ∈ [0,1] → grey hex."""
    v = max(0.0, min(1.0, value))
    g = int(lo + (hi - lo) * v)
    return f"#{g:02x}{g:02x}{g:02x}"


def main():
    config = load_config()
    update_step(config, STEP, status="running", progress_value=0.1,
                message="09a graph html")
    log_line("09a", "start")

    from pyvis.network import Network

    graph_pickle = resolve_path(config["paths"]["graph_pickle"])
    out_html = resolve_path("output/site/views/graph.html")
    ensure_parent(out_html)

    if not graph_pickle.exists():
        raise RuntimeError(f"graph pickle 없음: {graph_pickle}. 08 먼저 실행.")

    with open(graph_pickle, "rb") as f:
        G = pickle.load(f)

    log_line("09a", f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    # 필드 영문 description 로드 (tooltip용)
    corpus_path = resolve_path(config["paths"]["corpus_file"])
    descriptions: dict[str, str] = {}
    if corpus_path.exists():
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    descriptions[r["field_id"]] = r.get("description", "") or ""
                except Exception:
                    continue

    net = Network(
        height="100vh", width="100%",
        bgcolor="#0a0a0a", font_color="#f0f0f0",
        notebook=False, cdn_resources="in_line",
        directed=False,
    )
    net.barnes_hut(
        gravity=-16000, central_gravity=0.12,
        spring_length=140, spring_strength=0.015,
        damping=0.22,
    )

    # 노드 추가
    for n, d in G.nodes(data=True):
        is_centroid = bool(d.get("is_centroid", False))
        ac = int(d.get("alpha_count", 0) or 0)
        # 크기: centroid 크게, 나머지 alphaCount 로그
        import math
        size = 18 if is_centroid else 4 + min(12.0, math.log1p(ac))
        # 테두리 색: centroid 밝은 흰색, 나머지 중간 그레이
        border = "#ffffff" if is_centroid else "#6b6b6b"
        # 채움: 거의 투명한 검정
        bg = "#121212"

        desc = descriptions.get(n, "") or "(no description)"
        tooltip = f"{n}\n{desc}"
        net.add_node(
            n,
            label=n if is_centroid else "",
            size=size,
            color={"background": bg, "border": border,
                   "highlight": {"background": "#1a1a1a", "border": "#ffffff"}},
            borderWidth=1.5 if is_centroid else 1.0,
            title=tooltip,
            font={"color": "#f0f0f0", "size": 12, "face": "JetBrains Mono, monospace"},
            shape="dot",
        )

    # 엣지 추가
    for u, v, d in G.edges(data=True):
        kind = d.get("kind", "")
        sim = float(d.get("similarity", 0.0))
        color = _grey(sim)  # 유사도 높을수록 밝게
        width = 0.4 + sim * 2.2
        dashes = (kind == "far")
        title = (
            f"<b>{u} ↔ {v}</b><br>"
            f"similarity: {sim:.3f}<br>"
            f"distance: {1-sim:.3f}<br>"
            f"kind: {kind}"
        )
        net.add_edge(u, v, color=color, width=width, dashes=dashes, title=title)

    net.toggle_physics(True)
    # show_buttons 는 UI 가 지저분해지므로 끔. 필요하면 별도 control 로 제공.
    # net.show_buttons(filter_=["physics"])

    # 커스텀 options (배경/폰트 확실히 고정)
    net.set_options("""
    {
      "nodes": {
        "borderWidth": 1,
        "font": {
          "color": "#f0f0f0",
          "face": "JetBrains Mono, monospace",
          "size": 12
        }
      },
      "edges": {
        "smooth": { "type": "continuous" },
        "hoverWidth": 2
      },
      "interaction": {
        "hover": true,
        "hoverConnectedEdges": true,
        "selectConnectedEdges": true,
        "multiselect": false,
        "zoomView": true,
        "zoomSpeed": 0.6,
        "dragView": true,
        "keyboard": { "enabled": false }
      },
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -16000,
          "centralGravity": 0.12,
          "springLength": 140,
          "springConstant": 0.015,
          "damping": 0.22
        },
        "stabilization": { "iterations": 600 }
      }
    }
    """)

    out_html.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(out_html))

    # PyVis 가 생성한 HTML 은 기본 body background 가 흰색 — 강제 override
    html_text = out_html.read_text(encoding="utf-8")
    inject_style = """
    <style>
      html, body { background: #0a0a0a !important; margin: 0; padding: 0;
                   font-family: 'JetBrains Mono', monospace; }
      #mynetwork { background: #0a0a0a !important; }
      .vis-tooltip {
        background: #121212 !important;
        color: #f0f0f0 !important;
        border: 1px solid #2a2a2a !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px !important;
        padding: 10px 14px !important;
        border-radius: 2px !important;
        white-space: pre-line !important;
        max-width: 420px !important;
        line-height: 1.5 !important;
      }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    """
    html_text = html_text.replace("</head>", inject_style + "</head>")

    # 부모 → iframe postMessage 리스너 (검색 하이라이트 + 노란 마킹)
    inject_script = """
    <script>
    (function(){
      var prevId = null;
      var prevSnapshot = null;  // {color, borderWidth, size, font}
      var YELLOW = '#ffd600';
      var YELLOW_BORDER = '#ffea00';

      function revert(){
        if (prevId && prevSnapshot && typeof network !== 'undefined') {
          try {
            network.body.data.nodes.update({
              id: prevId,
              color: prevSnapshot.color,
              borderWidth: prevSnapshot.borderWidth,
              size: prevSnapshot.size,
              font: prevSnapshot.font
            });
          } catch(e){}
        }
        prevId = null; prevSnapshot = null;
      }

      function markYellow(fid){
        var node = network.body.data.nodes.get(fid);
        if (!node) return false;
        prevSnapshot = {
          color: node.color,
          borderWidth: node.borderWidth,
          size: node.size,
          font: node.font
        };
        network.body.data.nodes.update({
          id: fid,
          color: {
            background: YELLOW,
            border: YELLOW_BORDER,
            highlight: { background: YELLOW, border: YELLOW_BORDER },
            hover: { background: YELLOW, border: YELLOW_BORDER }
          },
          borderWidth: 4,
          size: (node.size || 15) + 8,
          font: { color: YELLOW, face: 'JetBrains Mono, monospace', size: 14, strokeWidth: 3, strokeColor: '#000000' }
        });
        prevId = fid;
        return true;
      }

      function tryFocus(fid, attempt){
        attempt = attempt || 0;
        if (typeof network === 'undefined') {
          if (attempt < 40) setTimeout(function(){ tryFocus(fid, attempt+1); }, 100);
          return;
        }
        try {
          revert();
          if (!markYellow(fid)) return;
          network.selectNodes([fid]);
          network.focus(fid, { scale: 1.8, animation: { duration: 500, easingFunction: 'easeInOutQuad' } });
        } catch(err) { console.warn('focus failed', err); }
      }

      window.addEventListener('message', function(e){
        var d = e.data || {};
        if (d.type === 'highlight' && d.field_id) tryFocus(d.field_id, 0);
        if (d.type === 'clear') {
          revert();
          try { network.unselectAll(); network.fit({ animation: true }); } catch(err){}
        }
      });
    })();
    </script>
    """
    html_text = html_text.replace("</body>", inject_script + "</body>")
    out_html.write_text(html_text, encoding="utf-8")

    log_line("09a", f"saved → {out_html}")
    update_step(config, STEP, progress_value=0.4,
                message="09a graph html done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
