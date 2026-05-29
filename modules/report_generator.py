"""
report_generator.py
-------------------
Build a one-page PDF report summarising inputs, predictions, and curves.
"""
from __future__ import annotations
import io
from datetime import datetime
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib import colors


def build_pdf(path: str, inputs: dict, results: dict, insights: str,
              curves: dict) -> str:
    """curves = {'slip': np.ndarray, 'gauss': np.ndarray, 'pacejka': np.ndarray}"""
    doc = SimpleDocTemplate(path, pagesize=A4)
    styles = getSampleStyleSheet()
    flow = []

    flow.append(Paragraph("Tyre Traction Simulation Report", styles["Title"]))
    flow.append(Paragraph(datetime.now().strftime("%Y-%m-%d %H:%M"), styles["Normal"]))
    flow.append(Spacer(1, 12))

    flow.append(Paragraph("<b>Inputs</b>", styles["Heading2"]))
    in_tbl = Table([[k, str(v)] for k, v in inputs.items()],
                   colWidths=[180, 280])
    in_tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey)]))
    flow.append(in_tbl)
    flow.append(Spacer(1, 10))

    flow.append(Paragraph("<b>Predictions</b>", styles["Heading2"]))
    res_tbl = Table([[k, f"{v:.4f}" if isinstance(v, (int, float)) else str(v)]
                     for k, v in results.items()], colWidths=[180, 280])
    res_tbl.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.4, colors.grey)]))
    flow.append(res_tbl)
    flow.append(Spacer(1, 10))

    # Plot curves
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(curves["slip"], curves["gauss"], "--", label="Gaussian")
    ax.plot(curves["slip"], curves["pacejka"], label="Pacejka")
    ax.set_xlabel("Slip ratio")
    ax.set_ylabel("µ")
    ax.set_title("µ vs Slip")
    ax.legend()
    ax.grid(alpha=0.3)
    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    flow.append(Image(buf, width=420, height=220))
    flow.append(Spacer(1, 10))

    flow.append(Paragraph("<b>Engineering Insights</b>", styles["Heading2"]))
    flow.append(Paragraph(insights, styles["Normal"]))

    doc.build(flow)
    return path
