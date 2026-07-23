"""
CrediLens AI - PDF Report Generator
======================================
Generates a professional, bank-style PDF decision report for a single
loan prediction. Used by app.py's "Download PDF Report" feature.

Requires: reportlab (pip install reportlab)
"""

from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import io

NAVY = colors.HexColor("#0A1F44")
GOLD = colors.HexColor("#D4AF37")
GREEN = colors.HexColor("#2E9E6B")
RED = colors.HexColor("#C73E3E")
LIGHT_GREY = colors.HexColor("#F4F6FA")


def generate_pdf_report(form: dict, prediction_label: str, prob_approved: float,
                         confidence: float, model_name: str, explanation_pairs=None) -> bytes:
    """Builds an in-memory PDF and returns its raw bytes."""

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=18 * mm, bottomMargin=18 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "TitleStyle", parent=styles["Title"], textColor=NAVY,
        fontSize=22, spaceAfter=2, alignment=TA_LEFT,
    )
    sub_style = ParagraphStyle(
        "SubStyle", parent=styles["Normal"], textColor=colors.grey,
        fontSize=10, spaceAfter=14,
    )
    section_style = ParagraphStyle(
        "SectionStyle", parent=styles["Heading2"], textColor=NAVY,
        fontSize=13, spaceBefore=14, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyStyle", parent=styles["Normal"], fontSize=10, leading=15,
    )

    elements = []
    elements.append(Paragraph("🏦 CrediLens AI", title_style))
    elements.append(Paragraph("Intelligent Loan Approval — Decision Report", sub_style))
    elements.append(HRFlowable(width="100%", color=GOLD, thickness=1.4))
    elements.append(Spacer(1, 10))

    # Decision banner
    decision_color = GREEN if prediction_label == "Approved" else RED
    decision_table = Table(
        [[f"DECISION:  {prediction_label.upper()}",
          f"Approval Probability:  {prob_approved*100:.1f}%"]],
        colWidths=[90 * mm, 80 * mm],
    )
    decision_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), decision_color),
        ("TEXTCOLOR", (0, 0), (-1, -1), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 12),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
    ]))
    elements.append(decision_table)
    elements.append(Spacer(1, 16))

    meta = [
        ["Report Generated", datetime.now().strftime("%d %b %Y, %H:%M:%S")],
        ["Model Engine", model_name],
        ["Model Confidence", f"{confidence:.2f}%"],
    ]
    meta_table = Table(meta, colWidths=[60 * mm, 110 * mm])
    meta_table.setStyle(TableStyle([
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.grey),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    elements.append(meta_table)

    elements.append(Paragraph("Applicant Details", section_style))
    applicant_rows = [
        ["Gender", form.get("gender", "-"), "Married", form.get("married", "-")],
        ["Dependents", form.get("dependents", "-"), "Education", form.get("education", "-")],
        ["Self-Employed", form.get("self_employed", "-"), "Property Area", form.get("property_area", "-")],
    ]
    applicant_table = Table(applicant_rows, colWidths=[35 * mm, 50 * mm, 35 * mm, 50 * mm])
    applicant_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_GREY),
        ("BACKGROUND", (2, 0), (2, -1), LIGHT_GREY),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(applicant_table)

    elements.append(Paragraph("Financial Details", section_style))
    fin_rows = [
        ["Applicant Income", f"Rs. {form.get('applicant_income', 0):,}",
         "Co-applicant Income", f"Rs. {form.get('coapplicant_income', 0):,}"],
        ["Loan Amount", f"Rs. {form.get('loan_amount', 0)}k",
         "Loan Term", f"{form.get('loan_term', '-')} months"],
        ["Credit History", "Good" if form.get("credit_history") else "Poor / None", "", ""],
    ]
    fin_table = Table(fin_rows, colWidths=[35 * mm, 50 * mm, 35 * mm, 50 * mm])
    fin_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), LIGHT_GREY),
        ("BACKGROUND", (2, 0), (2, -1), LIGHT_GREY),
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.lightgrey),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    elements.append(fin_table)

    if explanation_pairs:
        elements.append(Paragraph("AI Explanation — Key Decision Factors", section_style))
        for feat, val, _ in explanation_pairs:
            elements.append(Paragraph(f"&bull; <b>{feat}</b> — impact score: {abs(val):.3f}", body_style))

    elements.append(Spacer(1, 18))
    elements.append(HRFlowable(width="100%", color=colors.lightgrey, thickness=0.8))
    disclaimer = ParagraphStyle("Disclaimer", parent=styles["Normal"], fontSize=8,
                                 textColor=colors.grey, spaceBefore=8)
    elements.append(Paragraph(
        "This is an AI-generated risk assessment produced by CrediLens AI for demonstration "
        "and portfolio purposes. It does not constitute a final lending decision or financial advice.",
        disclaimer,
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()
