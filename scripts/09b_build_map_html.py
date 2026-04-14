"""
Step 09b — Plotly UMAP 2D 스캐터 HTML (모노크롬).

필드들의 combined distance 를 UMAP 로 2D 임베딩 → plotly scatter.
배경/점/축 전부 흰색-검정 계열만 사용.

출력:
  output/site/views/map.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, resolve_path, ensure_parent, log_line, update_step,
)

STEP = "09_visualize"


def main():
    config = load_config()
    log_line("09b", "start")

    data_dir = resolve_path(config["paths"]["data_dir"])
    sim_path = data_dir / "similarity_combined.npy"
    sim_ids_path = data_dir / "similarity_field_ids.json"
    clusters_path = data_dir / "clusters_behavior.json"
    corpus_path = resolve_path(config["paths"]["corpus_file"])
    meta_path = data_dir / "single_field_meta.jsonl"
    out_html = resolve_path("output/site/views/map.html")
    ensure_parent(out_html)

    combined = np.load(sim_path).astype(np.float32)
    field_ids = json.loads(sim_ids_path.read_text(encoding="utf-8"))
    clusters = json.loads(clusters_path.read_text(encoding="utf-8"))
    assignments = clusters["assignments"]
    centroid_set = set(clusters.get("centroids", {}).values())

    # 필드 메타
    field_meta = {}
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                field_meta[r["field_id"]] = r

    sim_meta = {}
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    sim_meta[r["field_id"]] = r

    # UMAP — distance = 1 - similarity
    try:
        import umap
    except ImportError:
        log_line("09b", "umap 없음 — fallback: PCA 2D")
        umap = None

    distance = 1.0 - combined
    np.fill_diagonal(distance, 0.0)
    distance = np.clip(distance, 0.0, None)
    distance = (distance + distance.T) / 2

    if umap is not None:
        reducer = umap.UMAP(
            n_components=2,
            metric='precomputed',
            n_neighbors=15,
            min_dist=0.1,
            random_state=42,
        )
        coords = reducer.fit_transform(distance.astype(np.float64))
    else:
        from sklearn.decomposition import PCA
        coords = PCA(n_components=2).fit_transform(combined)

    # 각 필드 데이터
    import pandas as pd
    rows = []
    for i, fid in enumerate(field_ids):
        m = field_meta.get(fid, {})
        sm = sim_meta.get(fid, {})
        rows.append({
            "field_id": fid,
            "x": float(coords[i, 0]),
            "y": float(coords[i, 1]),
            "cluster": str(assignments.get(fid, -1)),
            "category": m.get("category_name", ""),
            "subcategory": m.get("subcategory_name", ""),
            "alpha_count": int(m.get("alpha_count", 0)),
            "sharpe": float(sm.get("sharpe", 0.0)),
            "is_centroid": fid in centroid_set,
        })
    df = pd.DataFrame(rows)

    # 클러스터 -> 그레이 5단계
    clusters_unique = sorted(df["cluster"].unique(),
                             key=lambda c: (c == "-1", c))
    grey_levels = ["#6b6b6b", "#8a8a8a", "#a8a8a8", "#c8c8c8", "#ffffff"]
    cluster_color = {
        c: grey_levels[i % len(grey_levels)] if c != "-1" else "#3a3a3a"
        for i, c in enumerate(clusters_unique)
    }
    df["color"] = df["cluster"].map(cluster_color)
    df["size"] = df.apply(
        lambda r: 14 if r["is_centroid"] else 4 + min(10, np.log1p(r["alpha_count"])),
        axis=1,
    )

    import plotly.graph_objects as go

    fig = go.Figure()

    # 비센트로이드 점
    df_regular = df[~df["is_centroid"]]
    fig.add_trace(go.Scatter(
        x=df_regular["x"], y=df_regular["y"],
        mode="markers",
        marker=dict(
            size=df_regular["size"],
            color=df_regular["color"],
            line=dict(color="#1a1a1a", width=0.5),
            opacity=0.85,
        ),
        text=df_regular["field_id"],
        customdata=df_regular[["cluster", "category", "subcategory", "alpha_count", "sharpe"]],
        hovertemplate=(
            "<b>%{text}</b><br>"
            "cluster: %{customdata[0]}<br>"
            "category: %{customdata[1]}<br>"
            "subcategory: %{customdata[2]}<br>"
            "alpha_count: %{customdata[3]}<br>"
            "sharpe: %{customdata[4]:.2f}"
            "<extra></extra>"
        ),
        name="fields",
        showlegend=False,
    ))

    # 센트로이드 — 흰 테두리 링
    df_centroid = df[df["is_centroid"]]
    if len(df_centroid) > 0:
        fig.add_trace(go.Scatter(
            x=df_centroid["x"], y=df_centroid["y"],
            mode="markers+text",
            marker=dict(
                size=df_centroid["size"],
                color="#0a0a0a",
                line=dict(color="#ffffff", width=2),
            ),
            text=df_centroid["field_id"],
            textposition="top center",
            textfont=dict(color="#f0f0f0", family="JetBrains Mono", size=10),
            customdata=df_centroid[["cluster", "category", "subcategory", "alpha_count", "sharpe"]],
            hovertemplate=(
                "<b>%{text}</b> (centroid)<br>"
                "cluster: %{customdata[0]}<br>"
                "category: %{customdata[1]}<br>"
                "subcategory: %{customdata[2]}<br>"
                "alpha_count: %{customdata[3]}<br>"
                "sharpe: %{customdata[4]:.2f}"
                "<extra></extra>"
            ),
            name="centroids",
            showlegend=False,
        ))

    fig.update_layout(
        template=None,
        dragmode="pan",
        paper_bgcolor="#0a0a0a",
        plot_bgcolor="#0a0a0a",
        font=dict(color="#a8a8a8", family="JetBrains Mono, monospace", size=12),
        xaxis=dict(
            showgrid=False, zeroline=False,
            showticklabels=False, title="",
            linecolor="#2a2a2a",
        ),
        yaxis=dict(
            showgrid=False, zeroline=False,
            showticklabels=False, title="",
            linecolor="#2a2a2a",
        ),
        margin=dict(l=10, r=10, t=10, b=10),
        hoverlabel=dict(
            bgcolor="#121212",
            bordercolor="#2a2a2a",
            font=dict(color="#f0f0f0", family="JetBrains Mono"),
        ),
        showlegend=False,
    )

    html_str = fig.to_html(
        full_html=True,
        include_plotlyjs="cdn",
        config={"displayModeBar": False, "scrollZoom": True},
    )
    # body 배경 override
    html_str = html_str.replace(
        "<head>",
        "<head><style>html,body{background:#0a0a0a;margin:0;padding:0;"
        "font-family:'JetBrains Mono',monospace;}</style>"
        '<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">',
    )

    # UMAP 좌표 → 부모 main.js 가 사용할 수 있게 data_dir 에 dump
    coords_path = data_dir / "umap_coords.json"
    coords_json = {fid: [float(coords[i, 0]), float(coords[i, 1])]
                   for i, fid in enumerate(field_ids)}
    coords_path.write_text(json.dumps(coords_json, ensure_ascii=False),
                           encoding="utf-8")

    # postMessage 하이라이트 리스너 주입
    pos_js = json.dumps(coords_json, ensure_ascii=False)
    inject = """
    <script>
    (function(){
      var FIELD_POS = __POS__;
      var origRange = null;
      function findGd(){ return document.querySelector('.plotly-graph-div'); }
      function highlight(fid){
        var gd = findGd(); if (!gd) return;
        var pos = FIELD_POS[fid]; if (!pos) return;
        if (!origRange && gd.layout && gd.layout.xaxis) {
          origRange = {
            x: gd.layout.xaxis.range && gd.layout.xaxis.range.slice(),
            y: gd.layout.yaxis.range && gd.layout.yaxis.range.slice()
          };
        }
        var r = 1.8;
        Plotly.relayout(gd, {
          'xaxis.range': [pos[0] - r, pos[0] + r],
          'yaxis.range': [pos[1] - r, pos[1] + r]
        });
        var existing = (gd.data || []).findIndex(function(t){ return t.name === '__highlight__'; });
        var trace = {
          x: [pos[0]], y: [pos[1]],
          mode: 'markers+text',
          marker: { size: 22, color: '#ffd600',
                    line: { color: '#ffea00', width: 3 },
                    opacity: 1.0 },
          text: [fid],
          textposition: 'top center',
          textfont: { color: '#ffd600', family: 'JetBrains Mono', size: 11 },
          hoverinfo: 'text',
          name: '__highlight__',
          showlegend: false
        };
        if (existing >= 0) {
          Plotly.restyle(gd, {x: [[pos[0]]], y: [[pos[1]]], text: [[fid]]}, [existing]);
        } else {
          Plotly.addTraces(gd, [trace]);
        }
      }
      function clearFocus(){
        var gd = findGd(); if (!gd) return;
        if (origRange) {
          Plotly.relayout(gd, {
            'xaxis.range': origRange.x,
            'yaxis.range': origRange.y
          });
        }
        var idx = (gd.data || []).findIndex(function(t){ return t.name === '__highlight__'; });
        if (idx >= 0) Plotly.deleteTraces(gd, [idx]);
      }
      window.addEventListener('message', function(e){
        var d = e.data || {};
        if (d.type === 'highlight' && d.field_id) {
          var tryIt = function(n){
            if (findGd()) highlight(d.field_id);
            else if (n < 40) setTimeout(function(){ tryIt(n+1); }, 100);
          };
          tryIt(0);
        }
        if (d.type === 'clear') clearFocus();
      });
    })();
    </script>
    """.replace("__POS__", pos_js)
    html_str = html_str.replace("</body>", inject + "</body>")
    out_html.write_text(html_str, encoding="utf-8")

    log_line("09b", f"saved → {out_html}")
    update_step(config, STEP, progress_value=0.6,
                message="09b map html done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
