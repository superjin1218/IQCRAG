"""
Step 09 — PyVis 인터랙티브 시각화.

output/field_graph.gpickle 을 읽어서 HTML 하나로 떨어뜨린다. 브라우저에서 열면
드래그/줌/검색 가능한 그래프가 나온다.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    load_config, resolve_path, ensure_parent, update_step, log_line,
)

STEP = "09_visualize"


def main():
    config = load_config()
    update_step(config, STEP, status="running", progress_value=0.0, message="시작")
    log_line(STEP, "start")

    import networkx as nx
    from pyvis.network import Network

    graph_pickle = resolve_path(config["paths"]["graph_pickle"])
    out_html = resolve_path(config["paths"]["graph_html"])
    ensure_parent(out_html)

    if not graph_pickle.exists():
        raise RuntimeError(f"graph pickle 없음: {graph_pickle}. 08 단계 먼저 실행.")

    with open(graph_pickle, "rb") as f:
        G: nx.Graph = pickle.load(f)

    log_line(STEP, f"nodes={G.number_of_nodes()}, edges={G.number_of_edges()}")

    net = Network(
        height="900px", width="100%",
        bgcolor="#111111", font_color="#eeeeee",
        notebook=False, cdn_resources="in_line",
    )
    net.barnes_hut(
        gravity=-12000, central_gravity=0.25, spring_length=120, spring_strength=0.02,
    )

    # 노드
    for n, d in G.nodes(data=True):
        kind = d.get("kind", "member")
        is_rep = kind == "representative"
        size = 20 + (d.get("cluster_size", 1) or 1) * 0.5 if is_rep else 6
        color = "#4ec9b0" if is_rep else "#6a6a6a"
        title_lines = [
            f"<b>{n}</b>",
            f"kind: {kind}",
            f"cluster: {d.get('cluster', '')}",
        ]
        if is_rep:
            title_lines += [
                f"cluster_size: {d.get('cluster_size', '')}",
                f"alpha_count: {d.get('alpha_count', '')}",
                f"dataset: {d.get('dataset', '')}",
                f"category: {d.get('category', '')}",
            ]
        net.add_node(n, label=n if is_rep else "", size=size, color=color,
                     title="<br>".join(title_lines))

    # 엣지
    for u, v, d in G.edges(data=True):
        kind = d.get("kind", "")
        width = 1.0 + (float(d.get("weight", 0.5)) * 4.0 if kind == "inter" else 0.5)
        color = d.get("color", "#888888")
        title = ""
        if kind == "inter":
            title = (
                f"similarity: {d.get('similarity', 0):.3f}<br>"
                f"diversification: {d.get('diversification', 0):.3f}<br>"
                f"probe sharpe: {d.get('sharpe', 0):.2f}"
            )
        net.add_edge(u, v, width=width, color=color, title=title)

    # 컨트롤 패널 옵션 (간단)
    net.toggle_physics(True)
    net.show_buttons(filter_=["physics", "nodes", "edges"])

    out_html.parent.mkdir(parents=True, exist_ok=True)
    # PyVis save_graph 는 cwd 기준 — 절대경로로 강제
    net.save_graph(str(out_html))
    log_line(STEP, f"저장 → {out_html}")

    update_step(
        config, STEP, status="done", progress_value=1.0,
        message=f"HTML: {out_html.name}",
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        from common import load_config, update_step
        update_step(load_config(), STEP, status="failed", message=str(e)[:200])
        raise
