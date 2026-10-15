"""Render docs/proposal.md to docs/NABDH_Proposal.pdf.

Supports the Markdown subset the proposal uses: headings, paragraphs, bullet and
numbered lists, block quotes, pipe tables, images, **bold**, *italic*, `code`.

    python scripts/build_proposal_pdf.py
"""
import re
from pathlib import Path

import matplotlib
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (CondPageBreak, Image, KeepTogether, ListFlowable, ListItem, PageBreak,
                                Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "docs" / "proposal.md"
OUT = ROOT / "docs" / "NABDH_Proposal.pdf"

# DejaVu ships with matplotlib and covers Ω, ≥, ε and friends.
FONT_DIR = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(FONT_DIR / "DejaVuSansMono.ttf")))
pdfmetrics.registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-B")

BLUE = colors.HexColor("#2a78d6")
INK = colors.HexColor("#1d1d1b")
INK2 = colors.HexColor("#52514e")
RULE = colors.HexColor("#e4e3df")
TINT = colors.HexColor("#eef4fc")

S = {
    "title": ParagraphStyle("title", fontName="DV-B", fontSize=26, leading=32, textColor=INK,
                            spaceAfter=6),
    "subtitle": ParagraphStyle("subtitle", fontName="DV", fontSize=14, leading=19,
                               textColor=BLUE, spaceAfter=18),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=15, leading=19, textColor=INK,
                         spaceBefore=14, spaceAfter=6, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=12, leading=16, textColor=BLUE,
                         spaceBefore=10, spaceAfter=4, keepWithNext=1),
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.6, leading=14, textColor=INK,
                           spaceAfter=6, alignment=TA_LEFT),
    "meta": ParagraphStyle("meta", fontName="DV", fontSize=10.5, leading=16, textColor=INK2,
                           spaceAfter=4),
    "quote": ParagraphStyle("quote", fontName="DV-I", fontSize=9.4, leading=14, textColor=INK2,
                            leftIndent=12, borderPadding=(6, 8, 6, 8), backColor=TINT,
                            spaceBefore=4, spaceAfter=10),
    "lead": ParagraphStyle("lead", fontName="DV", fontSize=11.5, leading=17, textColor=INK,
                           borderPadding=(10, 12, 10, 12), backColor=TINT, leftIndent=12,
                           rightIndent=12, spaceBefore=6, spaceAfter=18),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=8.4, leading=11, textColor=INK),
    "cellh": ParagraphStyle("cellh", fontName="DV-B", fontSize=8.4, leading=11,
                            textColor=colors.white),
}


def inline(text: str) -> str:
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?!\w)", r"<i>\1</i>", text)
    text = re.sub(r"`(.+?)`", r'<font face="DVM" size="8.4">\1</font>', text)
    text = re.sub(r"\[(.+?)\]\((http[^)]+)\)", r'<link href="\2" color="#2a78d6">\1</link>', text)
    return text


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    data = [[Paragraph(inline(c), S["cellh"] if i == 0 else S["cell"]) for c in r]
            for i, r in enumerate(cells)]
    width = A4[0] - 40 * mm
    n = len(cells[0])
    t = Table(data, colWidths=[width / n] * n, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), BLUE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f5")]),
        ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return [t, Spacer(1, 8)]


def image(path: str):
    p = (SRC.parent / path).resolve()
    img = Image(str(p))
    max_w = A4[0] - 40 * mm
    scale = min(1.0, max_w / img.imageWidth)
    img.drawWidth, img.drawHeight = img.imageWidth * scale, img.imageHeight * scale
    if img.drawHeight > 115 * mm:
        k = 115 * mm / img.drawHeight
        img.drawWidth, img.drawHeight = img.drawWidth * k, img.drawHeight * k
    return [Spacer(1, 4), img, Spacer(1, 10)]


def build_story(md: str):
    story, para, lst, lst_kind, tbl = [], [], [], None, []
    first_h1_seen = False

    def flush():
        nonlocal para, lst, lst_kind, tbl
        if para:
            text = " ".join(para)
            if text.startswith("> "):
                style = S["quote"] if first_h1_seen else S["lead"]   # cover gets the big box
                story.append(Paragraph(inline(re.sub(r"^>\s?", "", text).replace(" > ", " ")),
                                       style))
            else:
                style = S["meta"] if not first_h1_seen else S["body"]
                story.append(Paragraph(inline(text), style))
            para = []
        if lst:
            items = [ListItem(Paragraph(inline(x), S["body"]), leftIndent=12) for x in lst]
            ordered = lst_kind == "ol"
            story.append(ListFlowable(items, bulletType="1" if ordered else "bullet",
                                      start="1" if ordered else "•", bulletFontName="DV",
                                      bulletFontSize=8, leftIndent=14, bulletColor=BLUE))
            lst, lst_kind = [], None
        if tbl:
            story.extend(table(tbl))
            tbl = []

    for line in md.splitlines():
        s = line.rstrip()
        if not s.strip():
            flush(); continue
        if s.startswith("# "):
            flush(); story.append(Spacer(1, 6 * mm)); story.append(Paragraph(inline(s[2:]), S["title"]))
            continue
        if s.startswith("## "):
            flush()
            if not first_h1_seen and not re.match(r"## \d", s):
                story.append(Paragraph(inline(s[3:]), S["subtitle"]))
                continue
            if not first_h1_seen:
                story.append(PageBreak())
            first_h1_seen = True
            story.append(CondPageBreak(45 * mm))   # never strand a section title at a page foot
            story.append(Paragraph(inline(s[3:]), S["h1"]))
            continue
        if s.startswith("### "):
            flush(); story.append(CondPageBreak(30 * mm))
            story.append(Paragraph(inline(s[4:]), S["h2"])); continue
        m = re.match(r"!\[.*?\]\((.+?)\)", s.strip())
        if m:
            flush(); story.append(KeepTogether(image(m.group(1)))); continue
        if s.lstrip().startswith("|"):
            if para or lst:
                flush()
            tbl.append(s); continue
        m = re.match(r"\s*(-|\d+\.)\s+(.*)", s)
        if m and not para:
            kind = "ol" if m.group(1)[0].isdigit() else "ul"
            if lst and kind != lst_kind:
                flush()
            lst.append(m.group(2)); lst_kind = kind; continue
        if lst and s.startswith("  "):
            lst[-1] += " " + s.strip(); continue
        para.append(s.strip())
    flush()
    return story


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(INK2)
    canvas.drawString(20 * mm, 12 * mm, "NABDH | TSYP14 EnergyMind Tunisia | Phase 1 proposal")
    canvas.drawRightString(A4[0] - 20 * mm, 12 * mm, str(doc.page))
    canvas.restoreState()


def main():
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=18 * mm, bottomMargin=20 * mm,
                            title="NABDH: the grid signs its own data",
                            author="NABDH project", subject="TSYP14 EnergyMind Tunisia, Phase 1")
    doc.build(build_story(SRC.read_text(encoding="utf-8")), onFirstPage=footer, onLaterPages=footer)
    print("wrote", OUT.relative_to(ROOT))


if __name__ == "__main__":
    main()
