import plotly.graph_objects as go
import plotly.io as pio
import colorcet as cc


pio.templates["cc"] = go.layout.Template(
    data=dict(
        scatter=[
            dict(marker=dict(size=10, line=dict(width=0.5, color="DarkSlateGrey")))
        ],
        scatter3d=[
            dict(marker=dict(line=dict(width=0.5, color="DarkSlateGrey")))
        ]
    )
).update(
    layout_height=400,
    layout_colorway=cc.glasbey[:40],
    layout_paper_bgcolor="#fafbfa",
    layout_plot_bgcolor="#fafbfa",
    layout_scene_xaxis_backgroundcolor="#fafbfa",
    layout_scene_yaxis_backgroundcolor="#fafbfa",
    layout_scene_zaxis_backgroundcolor="#fafbfa",
)

def save(fig, name):
    fig.write_html(
        f"html/{name}.html",
        config={"responsive": True},
        full_html=False,
        include_plotlyjs=False,
    )
    fig.write_json(f"json/{name}.json", pretty=False)
    return fig