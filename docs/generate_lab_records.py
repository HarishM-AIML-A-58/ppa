"""
generate_lab_records.py
-----------------------
Converts all Markdown lab records into college-format Word (.docx) documents.

College Lab Record format includes:
  - Cover / title block (Experiment No., Name, Date, Subject)
  - Aim
  - Objective
  - Dataset Description  (table)
  - Methodology          (numbered list)
  - Observations         (bullet list)
  - Inference            (bullet list)
  - Prescriptive Insight (table or bullet list)
  - Result
  - Output files note
  - Signature block

Usage:
    python docs/generate_lab_records.py
"""

from __future__ import annotations

import re
from pathlib import Path
from datetime import date

from docx import Document
from docx.shared import Pt, Cm, RGBColor, Inches
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT   = Path(__file__).resolve().parents[1]
MD_DIR      = REPO_ROOT / "docs" / "lab_records"
DOCX_DIR    = REPO_ROOT / "docs" / "lab_records"

COLLEGE_NAME = "Department of Artificial Intelligence & Machine Learning"
SUBJECT      = "Predictive and Prescriptive Analytics Laboratory"
SUBJECT_CODE = "21AIL76"
SEMESTER     = "VII Semester"
TODAY        = date.today().strftime("%d %B %Y")

# Experiment metadata (title, short name)
EXP_META = {
    "exp1_clustering":           ("Experiment 1", "Customer Segmentation via Clustering"),
    "exp2_statistics":           ("Experiment 2", "Statistical Analysis & Hypothesis Testing"),
    "exp3_data_cleaning":        ("Experiment 3", "Data Cleaning & Quality Assessment"),
    "exp4_visualization":        ("Experiment 4", "Data Visualization & Pattern Discovery"),
    "exp5_feature_engineering":  ("Experiment 5", "Feature Engineering for Predictive Modeling"),
    "exp6_association_rules":    ("Experiment 6", "Association Rules Mining — Market Basket Analysis"),
    "exp7_regression_models":    ("Experiment 7", "Regression Models — Housing Price Prediction"),
    "exp8_classification_models":("Experiment 8", "Classification Models — Credit Card Fraud Detection"),
    "exp9_temporal_forecasting": ("Experiment 9", "Temporal Forecasting — Stock Price Prediction"),
    "exp10_microarray":          ("Experiment 10","Microarray Analysis — Cancer Gene Expression Classification"),
}


# ── Helper: formatting ─────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color: str) -> None:
    """Set table cell background colour."""
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color.upper())
    tcPr.append(shd)


def set_row_height(row, height_cm: float) -> None:
    tr   = row._tr
    trPr = tr.get_or_add_trPr()
    trH  = OxmlElement("w:trHeight")
    trH.set(qn("w:val"),  str(int(height_cm * 567)))   # 1 cm ≈ 567 twips
    trH.set(qn("w:hRule"), "atLeast")
    trPr.append(trH)


def add_paragraph(doc: Document, text: str = "", style: str = "Normal",
                  bold: bool = False, italic: bool = False,
                  font_size: int = 11, color: str | None = None,
                  align: str = "LEFT", space_before: int = 0,
                  space_after: int = 4) -> None:
    para = doc.add_paragraph(style=style)
    para.alignment = {
        "LEFT":   WD_ALIGN_PARAGRAPH.LEFT,
        "CENTER": WD_ALIGN_PARAGRAPH.CENTER,
        "RIGHT":  WD_ALIGN_PARAGRAPH.RIGHT,
        "JUSTIFY":WD_ALIGN_PARAGRAPH.JUSTIFY,
    }.get(align, WD_ALIGN_PARAGRAPH.LEFT)
    para.paragraph_format.space_before = Pt(space_before)
    para.paragraph_format.space_after  = Pt(space_after)
    if text:
        run = para.add_run(text)
        run.bold      = bold
        run.italic    = italic
        run.font.size = Pt(font_size)
        if color:
            r, g, b = int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)
            run.font.color.rgb = RGBColor(r, g, b)
    return para


def add_section_heading(doc: Document, title: str) -> None:
    """Dark navy section heading bar."""
    para = doc.add_paragraph()
    para.paragraph_format.space_before = Pt(8)
    para.paragraph_format.space_after  = Pt(4)
    run = para.add_run(f"  {title.upper()}")
    run.bold      = True
    run.font.size = Pt(12)
    run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
    # Background shading via paragraph XML
    pPr = para._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  "1F3864")
    pPr.append(shd)


def add_simple_table(doc: Document, rows: list[list[str]],
                     header: bool = True,
                     col_widths: list[float] | None = None) -> None:
    """Add a bordered table. First row is header if header=True."""
    if not rows:
        return
    n_cols = len(rows[0])
    tbl = doc.add_table(rows=len(rows), cols=n_cols)
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Column widths
    if col_widths:
        for i, w in enumerate(col_widths):
            for row in tbl.rows:
                row.cells[i].width = Cm(w)

    for r_idx, row_data in enumerate(rows):
        row = tbl.rows[r_idx]
        set_row_height(row, 0.75)
        for c_idx, cell_text in enumerate(row_data):
            cell = row.cells[c_idx]
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            para = cell.paragraphs[0]
            para.alignment = WD_ALIGN_PARAGRAPH.LEFT
            para.paragraph_format.space_before = Pt(1)
            para.paragraph_format.space_after  = Pt(1)
            run = para.add_run(cell_text)
            run.font.size = Pt(10)
            if r_idx == 0 and header:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                set_cell_bg(cell, "2E74B5")
            elif r_idx % 2 == 0 and not (r_idx == 0 and header):
                set_cell_bg(cell, "EEF3FA")


def add_numbered_list(doc: Document, items: list[str]) -> None:
    for i, item in enumerate(items, 1):
        # Bold any **text** markers
        para = doc.add_paragraph(style="Normal")
        para.paragraph_format.left_indent  = Cm(1)
        para.paragraph_format.space_before = Pt(1)
        para.paragraph_format.space_after  = Pt(3)
        # Render inline bold (**text**)
        _render_inline(para, f"{i}.  {item}")


def add_bullet_list(doc: Document, items: list[str]) -> None:
    for item in items:
        para = doc.add_paragraph(style="Normal")
        para.paragraph_format.left_indent  = Cm(1)
        para.paragraph_format.space_before = Pt(1)
        para.paragraph_format.space_after  = Pt(3)
        _render_inline(para, f"•  {item}")


def _render_inline(para, text: str) -> None:
    """Render text with **bold** markers as bold runs."""
    parts = re.split(r"(\*\*[^*]+\*\*)", text)
    for part in parts:
        if part.startswith("**") and part.endswith("**"):
            run = para.add_run(part[2:-2])
            run.bold      = True
            run.font.size = Pt(10.5)
        else:
            run = para.add_run(part)
            run.font.size = Pt(10.5)


# ── Markdown parser ────────────────────────────────────────────────────────────

def parse_md(md_path: Path) -> dict:
    """
    Parse a lab-record Markdown file into structured sections.

    Returns dict with keys:
        title, aim, objective, dataset_rows, methodology,
        observations, inference, prescriptive, result, output_note
    """
    text = md_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    sections: dict = {
        "title":        "",
        "aim":          "",
        "objective":    [],
        "dataset_rows": [],   # list of [attr, value]
        "methodology":  [],
        "observations": [],
        "inference":    [],
        "prescriptive": [],   # list of strings or [list-of-row-lists]
        "prescriptive_table": [],
        "result":       [],
        "output_note":  "",
    }

    current_section = None
    in_table        = False
    table_rows: list[list[str]] = []

    def flush_table():
        nonlocal table_rows, in_table
        if table_rows:
            # Decide which section gets the table
            if current_section == "dataset_description":
                # Strip bold markers from first column
                for row in table_rows:
                    row[0] = row[0].replace("**", "")
                sections["dataset_rows"] = table_rows
            elif current_section == "prescriptive_insight":
                sections["prescriptive_table"] = table_rows
        table_rows = []
        in_table   = False

    def clean(s: str) -> str:
        # Remove trailing code-backtick spans but keep text
        s = re.sub(r"`([^`]+)`", r"\1", s)
        s = s.strip()
        return s

    i = 0
    while i < len(lines):
        line = lines[i]

        # H1 → title
        if line.startswith("# "):
            sections["title"] = line[2:].strip()
            i += 1
            continue

        # H2 → section switch
        if line.startswith("## "):
            if in_table:
                flush_table()
            sec = line[3:].strip().lower().replace(" ", "_").replace("&", "and")
            sec = re.sub(r"[^a-z_]", "", sec)
            current_section = sec
            i += 1
            continue

        # Table row
        if line.startswith("|") and current_section in (
                "dataset_description", "prescriptive_insight"):
            in_table = True
            cells = [c.strip() for c in line.strip("|").split("|")]
            # Skip separator rows (---|---)
            if all(re.match(r"^[-: ]+$", c) for c in cells if c):
                i += 1
                continue
            if cells:
                table_rows.append(cells)
            i += 1
            continue
        else:
            if in_table:
                flush_table()

        # Bullet / numbered list items
        stripped = line.strip()
        if stripped.startswith(("- ", "* ", "+ ")):
            item = clean(stripped[2:])
            if current_section == "objective":
                sections["objective"].append(item)
            elif current_section == "observations":
                sections["observations"].append(item)
            elif current_section == "inference":
                sections["inference"].append(item)
            elif current_section == "prescriptive_insight":
                sections["prescriptive"].append(item)
            elif current_section == "result":
                sections["result"].append(item)
            i += 1
            continue

        if re.match(r"^\d+\.\s", stripped):
            item = clean(re.sub(r"^\d+\.\s+", "", stripped))
            if current_section == "methodology":
                sections["methodology"].append(item)
            i += 1
            continue

        # Aim / plain paragraph text
        if current_section == "aim" and stripped and not stripped.startswith("#"):
            if sections["aim"]:
                sections["aim"] += " " + clean(stripped)
            else:
                sections["aim"] = clean(stripped)

        # Output note (italic line at bottom)
        if stripped.startswith("*Output files:"):
            sections["output_note"] = clean(stripped.strip("*"))

        i += 1

    if in_table:
        flush_table()

    return sections


# ── DOCX builder ───────────────────────────────────────────────────────────────

def build_docx(sections: dict, exp_key: str, output_path: Path) -> None:
    exp_no, exp_name = EXP_META.get(exp_key, ("Experiment", sections.get("title", "")))

    doc = Document()

    # ── Page margins ──────────────────────────────────────────────────────────
    for section in doc.sections:
        section.top_margin    = Cm(2.0)
        section.bottom_margin = Cm(2.0)
        section.left_margin   = Cm(2.5)
        section.right_margin  = Cm(2.0)

    # ── Default paragraph font ────────────────────────────────────────────────
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(11)

    # ══════════════════════════════════════════════════════════════════════════
    # 1. COLLEGE HEADER BLOCK
    # ══════════════════════════════════════════════════════════════════════════
    header_tbl = doc.add_table(rows=1, cols=1)
    header_tbl.style = "Table Grid"
    header_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    hcell = header_tbl.rows[0].cells[0]
    set_cell_bg(hcell, "1F3864")

    def hline(text: str, sz: int = 13, bold: bool = False) -> None:
        p = hcell.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after  = Pt(2)
        r = p.add_run(text)
        r.font.size  = Pt(sz)
        r.font.bold  = bold
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)

    hline(COLLEGE_NAME,   sz=13, bold=True)
    hline(SUBJECT,        sz=12, bold=True)
    hline(f"{SUBJECT_CODE}  |  {SEMESTER}", sz=11)

    doc.add_paragraph()   # spacing

    # ══════════════════════════════════════════════════════════════════════════
    # 2. EXPERIMENT TITLE BLOCK
    # ══════════════════════════════════════════════════════════════════════════
    title_tbl = doc.add_table(rows=3, cols=2)
    title_tbl.style = "Table Grid"
    title_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER

    def title_row(row_idx: int, label: str, value: str,
                  bold_val: bool = False, span: bool = False) -> None:
        row = title_tbl.rows[row_idx]
        set_row_height(row, 0.9)
        set_cell_bg(row.cells[0], "D6E4F7")
        lrun = row.cells[0].paragraphs[0].add_run(label)
        lrun.bold = True; lrun.font.size = Pt(11)
        row.cells[0].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.CENTER

        vrun = row.cells[1].paragraphs[0].add_run(value)
        vrun.bold = bold_val; vrun.font.size = Pt(11)
        row.cells[1].paragraphs[0].alignment = WD_ALIGN_PARAGRAPH.LEFT
        row.cells[1].paragraphs[0].paragraph_format.left_indent = Cm(0.3)

    title_row(0, "Experiment No.",  exp_no,   bold_val=True)
    title_row(1, "Experiment Name", exp_name, bold_val=True)
    title_row(2, "Date",            TODAY)
    # Make Experiment Name cell span value wider
    for row in title_tbl.rows:
        row.cells[0].width = Cm(4.5)
        row.cells[1].width = Cm(13)

    doc.add_paragraph()   # spacing

    # ══════════════════════════════════════════════════════════════════════════
    # 3. AIM
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Aim")
    p = doc.add_paragraph(style="Normal")
    p.paragraph_format.left_indent  = Cm(0.5)
    p.paragraph_format.space_after  = Pt(6)
    _render_inline(p, sections.get("aim", ""))

    # ══════════════════════════════════════════════════════════════════════════
    # 4. OBJECTIVE
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Objective")
    obj = sections.get("objective", [])
    if obj:
        add_numbered_list(doc, obj)
    else:
        add_paragraph(doc, "Refer to experiment description.", font_size=11)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. DATASET DESCRIPTION
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Dataset Description")
    ds_rows = sections.get("dataset_rows", [])
    if ds_rows:
        full = [["Attribute", "Description"]] + ds_rows
        add_simple_table(doc, full, header=True, col_widths=[4.5, 13])
    else:
        add_paragraph(doc, "Dataset details are described in the methodology section.", font_size=11)

    doc.add_paragraph()

    # ══════════════════════════════════════════════════════════════════════════
    # 6. METHODOLOGY
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Methodology")
    meth = sections.get("methodology", [])
    if meth:
        add_numbered_list(doc, meth)
    else:
        add_paragraph(doc, "See code implementation.", font_size=11)

    # ══════════════════════════════════════════════════════════════════════════
    # 7. OBSERVATIONS
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Observations")
    obs = sections.get("observations", [])
    if obs:
        add_bullet_list(doc, obs)
    else:
        add_paragraph(doc, "Observations recorded during experiment execution.", font_size=11)

    # ══════════════════════════════════════════════════════════════════════════
    # 8. INFERENCE
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Inference")
    inf = sections.get("inference", [])
    if inf:
        add_bullet_list(doc, inf)
    else:
        add_paragraph(doc, "Inferences drawn from experimental results.", font_size=11)

    # ══════════════════════════════════════════════════════════════════════════
    # 9. PRESCRIPTIVE INSIGHT
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Prescriptive Insight")
    pr_table = sections.get("prescriptive_table", [])
    pr_list  = sections.get("prescriptive", [])

    if pr_table:
        add_simple_table(doc, pr_table, header=True)
    if pr_list:
        add_bullet_list(doc, pr_list)
    if not pr_table and not pr_list:
        add_paragraph(doc, "Prescriptive actions derived from model outputs.", font_size=11)

    doc.add_paragraph()

    # ══════════════════════════════════════════════════════════════════════════
    # 10. RESULT
    # ══════════════════════════════════════════════════════════════════════════
    add_section_heading(doc, "Result")
    res = sections.get("result", [])
    if res:
        add_bullet_list(doc, res)
    else:
        add_paragraph(doc, "Results saved to outputs/ directory.", font_size=11)

    # Output files note
    note = sections.get("output_note", "")
    if note:
        p = doc.add_paragraph(style="Normal")
        p.paragraph_format.left_indent = Cm(0.5)
        p.paragraph_format.space_before = Pt(4)
        r = p.add_run(note)
        r.italic    = True
        r.font.size = Pt(9.5)
        r.font.color.rgb = RGBColor(0x44, 0x44, 0x77)

    # ══════════════════════════════════════════════════════════════════════════
    # 11. SIGNATURE BLOCK
    # ══════════════════════════════════════════════════════════════════════════
    doc.add_paragraph()
    sig_tbl = doc.add_table(rows=1, cols=3)
    sig_tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    labels = ["Student Signature", "Date of Submission", "Faculty Signature"]
    for i, lbl in enumerate(labels):
        c = sig_tbl.rows[0].cells[i]
        set_row_height(sig_tbl.rows[0], 1.8)
        set_cell_bg(c, "F0F4FC")
        p = c.paragraphs[0]
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_before = Pt(30)   # blank space for signature
        r = p.add_run(f"\n\n{'_' * 20}\n{lbl}")
        r.font.size = Pt(10)
        r.bold      = True

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_path))
    print(f"  ✓  Saved: {output_path.relative_to(REPO_ROOT)}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Generating College Lab Records (.docx)")
    print("=" * 60)

    # Map stem → exp_key
    stem_to_key = {
        "exp1_clustering":            "exp1_clustering",
        "exp1_lab_record":            "exp1_clustering",
        "exp2_statistics":            "exp2_statistics",
        "exp3_data_cleaning":         "exp3_data_cleaning",
        "exp4_visualization":         "exp4_visualization",
        "exp5_feature_engineering":   "exp5_feature_engineering",
        "exp6_association_rules":     "exp6_association_rules",
        "exp7_regression_models":     "exp7_regression_models",
        "exp7_lab_record":            "exp7_regression_models",
        "exp8_classification_models": "exp8_classification_models",
        "exp9_temporal_forecasting":  "exp9_temporal_forecasting",
        "exp10_microarray":           "exp10_microarray",
    }

    # Prefer the newer named files; skip old *_lab_record.md if new version exists
    newer_stems = {
        "exp1_lab_record":  "exp1_clustering",
        "exp7_lab_record":  "exp7_regression_models",
    }

    processed: set[str] = set()
    md_files = sorted(MD_DIR.glob("*.md"))

    for md_path in md_files:
        stem = md_path.stem
        exp_key = stem_to_key.get(stem)
        if exp_key is None:
            print(f"  [skip] {md_path.name} (no mapping)")
            continue
        if exp_key in processed:
            print(f"  [skip] {md_path.name} (duplicate, already generated)")
            continue

        out_name = exp_key + ".docx"
        out_path = DOCX_DIR / out_name

        try:
            sections = parse_md(md_path)
            build_docx(sections, exp_key, out_path)
            processed.add(exp_key)
        except Exception as exc:
            print(f"  [ERROR] {md_path.name}: {exc}")
            import traceback; traceback.print_exc()

    print(f"\nDone — {len(processed)} documents generated in {DOCX_DIR.relative_to(REPO_ROOT)}/\n")


if __name__ == "__main__":
    main()
