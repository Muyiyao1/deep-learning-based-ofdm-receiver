from __future__ import annotations

import argparse
import html
from pathlib import Path
import re
import sys

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    Preformatted,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
INLINE_CODE_RE = re.compile(r"`([^`]+)`")
IMAGE_RE = re.compile(r"^!\[([^\]]*)\]\(([^)]+)\)\s*$")
HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
NUMBER_RE = re.compile(r"^\d+\.\s+(.*)$")


def register_cjk_font() -> None:
    candidates = [
        Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
        Path(r"C:\Windows\Fonts\simhei.ttf"),
        Path(r"C:\Windows\Fonts\Deng.ttf"),
    ]
    font_path = next((path for path in candidates if path.exists()), None)
    if font_path is None:
        raise FileNotFoundError("No Chinese font found in C:\\Windows\\Fonts.")
    pdfmetrics.registerFont(TTFont("CJK", str(font_path)))
    pdfmetrics.registerFont(TTFont("CJKBold", str(font_path)))


def make_styles() -> dict[str, ParagraphStyle]:
    styles = getSampleStyleSheet()
    base = ParagraphStyle(
        "CJKBody",
        parent=styles["BodyText"],
        fontName="CJK",
        fontSize=10.2,
        leading=16,
        alignment=TA_LEFT,
        wordWrap="CJK",
        spaceAfter=5,
    )
    return {
        "base": base,
        "title": ParagraphStyle(
            "CJKTitle",
            parent=base,
            fontName="CJKBold",
            fontSize=22,
            leading=28,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#10233F"),
            spaceAfter=16,
        ),
        "h1": ParagraphStyle(
            "CJKH1",
            parent=base,
            fontName="CJKBold",
            fontSize=16,
            leading=22,
            textColor=colors.HexColor("#173A5E"),
            spaceBefore=12,
            spaceAfter=8,
        ),
        "h2": ParagraphStyle(
            "CJKH2",
            parent=base,
            fontName="CJKBold",
            fontSize=13,
            leading=18,
            textColor=colors.HexColor("#245B78"),
            spaceBefore=8,
            spaceAfter=6,
        ),
        "h3": ParagraphStyle(
            "CJKH3",
            parent=base,
            fontName="CJKBold",
            fontSize=11.5,
            leading=17,
            textColor=colors.HexColor("#334155"),
            spaceBefore=6,
            spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "CJKBullet",
            parent=base,
            leftIndent=16,
            firstLineIndent=-8,
            bulletIndent=0,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=styles["Code"],
            fontName="Courier",
            fontSize=8.2,
            leading=10.5,
            backColor=colors.HexColor("#F5F7FA"),
            borderColor=colors.HexColor("#D8DEE9"),
            borderWidth=0.5,
            borderPadding=6,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=base,
            fontSize=9,
            leading=12,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#475569"),
        ),
    }


def clean_inline(text: str) -> str:
    parts: list[str] = []
    pos = 0
    for match in INLINE_CODE_RE.finditer(text):
        parts.append(html.escape(text[pos : match.start()]))
        parts.append(f'<font name="Courier">{html.escape(match.group(1))}</font>')
        pos = match.end()
    parts.append(html.escape(text[pos:]))
    return "".join(parts)


def is_table_line(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|")


def is_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def make_table(lines: list[str], width: float, style: ParagraphStyle) -> Table | None:
    rows = []
    for line in lines:
        if is_separator(line):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        rows.append([Paragraph(clean_inline(cell), style) for cell in cells])
    if not rows:
        return None
    cols = max(len(row) for row in rows)
    for row in rows:
        while len(row) < cols:
            row.append(Paragraph("", style))
    table = Table(rows, colWidths=[width / cols] * cols, repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), "CJK"),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#E8F0F7")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#10233F")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CBD5E1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def make_image(path_text: str, alt: str, width: float, caption_style: ParagraphStyle, base_style: ParagraphStyle):
    image_path = (ROOT / path_text).resolve()
    if not image_path.exists():
        return Paragraph(f"[Missing image: {html.escape(path_text)}]", base_style)
    image = Image(str(image_path))
    scale = min(width / image.drawWidth, (13.5 * cm) / image.drawHeight, 1.0)
    image.drawWidth *= scale
    image.drawHeight *= scale
    return KeepTogether(
        [
            image,
            Spacer(1, 0.15 * cm),
            Paragraph(clean_inline(alt or image_path.name), caption_style),
            Spacer(1, 0.25 * cm),
        ]
    )


def flush_paragraph(buffer: list[str], story: list, style: ParagraphStyle) -> None:
    if not buffer:
        return
    text = " ".join(item.strip() for item in buffer if item.strip())
    if text:
        story.append(Paragraph(clean_inline(text), style))
    buffer.clear()


def build_story(markdown: str, width: float, styles: dict[str, ParagraphStyle]) -> list:
    story: list = []
    lines = markdown.splitlines()
    buffer: list[str] = []
    i = 0
    first_title = True

    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped:
            flush_paragraph(buffer, story, styles["base"])
            # Avoid a trailing spacer that can force a fully blank final page.
            if any(line.strip() for line in lines[i + 1 :]):
                story.append(Spacer(1, 0.08 * cm))
            i += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph(buffer, story, styles["base"])
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1
            if i < len(lines):
                i += 1
            story.append(Preformatted("\n".join(code_lines), styles["code"], maxLineLength=90))
            if i < len(lines):
                story.append(Spacer(1, 0.18 * cm))
            continue

        if is_table_line(stripped):
            flush_paragraph(buffer, story, styles["base"])
            table_lines = []
            while i < len(lines) and is_table_line(lines[i].strip()):
                table_lines.append(lines[i])
                i += 1
            table = make_table(table_lines, width, styles["base"])
            if table is not None:
                story.append(table)
                story.append(Spacer(1, 0.22 * cm))
            continue

        image_match = IMAGE_RE.match(stripped)
        if image_match:
            flush_paragraph(buffer, story, styles["base"])
            story.append(make_image(image_match.group(2), image_match.group(1), width, styles["caption"], styles["base"]))
            i += 1
            continue

        heading_match = HEADING_RE.match(stripped)
        if heading_match:
            flush_paragraph(buffer, story, styles["base"])
            marks, text = heading_match.groups()
            if len(marks) == 1 and first_title:
                story.append(Paragraph(clean_inline(text), styles["title"]))
                first_title = False
            elif len(marks) <= 2:
                story.append(Paragraph(clean_inline(text), styles["h1"]))
            elif len(marks) == 3:
                story.append(Paragraph(clean_inline(text), styles["h2"]))
            else:
                story.append(Paragraph(clean_inline(text), styles["h3"]))
            i += 1
            continue

        if stripped.startswith("- "):
            flush_paragraph(buffer, story, styles["base"])
            story.append(Paragraph(clean_inline(stripped[2:].strip()), styles["bullet"], bulletText="-"))
            i += 1
            continue

        if NUMBER_RE.match(stripped):
            flush_paragraph(buffer, story, styles["base"])
            story.append(Paragraph(clean_inline(stripped), styles["bullet"]))
            i += 1
            continue

        buffer.append(stripped)
        i += 1

    flush_paragraph(buffer, story, styles["base"])
    return story


def add_page(canvas, doc) -> None:
    canvas.saveState()
    width, _ = A4
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(colors.HexColor("#64748B"))
    canvas.drawString(doc.leftMargin, 1.1 * cm, "Deep Learning Based OFDM Receiver")
    canvas.drawRightString(width - doc.rightMargin, 1.1 * cm, f"Page {doc.page}")
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.setLineWidth(0.3)
    canvas.line(doc.leftMargin, 1.45 * cm, width - doc.rightMargin, 1.45 * cm)
    canvas.restoreState()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a Chinese README PDF with embedded result images.")
    parser.add_argument("--input", default="README.zh-CN.md", help="Markdown README path.")
    parser.add_argument("--output", default="report.pdf", help="Output PDF path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    register_cjk_font()
    styles = make_styles()
    input_path = (ROOT / args.input).resolve()
    output_path = (ROOT / args.output).resolve()
    markdown = input_path.read_text(encoding="utf-8")
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=1.65 * cm,
        leftMargin=1.65 * cm,
        topMargin=1.55 * cm,
        bottomMargin=1.85 * cm,
        title="深度学习辅助的 OFDM 信道估计：公平基线、物理先验与泛化分析",
        author="Deep Learning Based OFDM Receiver",
    )
    doc.build(build_story(markdown, doc.width, styles), onFirstPage=add_page, onLaterPages=add_page)
    print(output_path)


if __name__ == "__main__":
    try:
        main()
    except PermissionError as exc:
        print(f"Permission error while writing PDF: {exc}", file=sys.stderr)
        print("Close the PDF viewer that is holding the output file, then run this script again.", file=sys.stderr)
        raise
