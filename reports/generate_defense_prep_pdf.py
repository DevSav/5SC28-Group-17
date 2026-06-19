import re
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer


ROOT = Path(__file__).resolve().parent
TEX_FILE = ROOT / "defense_prep_analytical_report.tex"
PDF_FILE = ROOT / "defense_prep_analytical_report.pdf"


def clean_latex_text(text):
    text = text.replace(r"\textbf{", "")
    text = text.replace(r"\textit{", "")
    text = text.replace(r"\texttt{", "")
    text = text.replace(r"\mathrm{", "")
    text = text.replace(r"\theta", "theta")
    text = text.replace(r"\Delta", "Delta")
    text = text.replace(r"\hat", "hat")
    text = text.replace(r"\pi", "pi")
    text = text.replace(r"\rightarrow", "->")
    text = text.replace(r"\approx", "~")
    text = text.replace(r"\subsection*", "")
    text = text.replace(r"\section*", "")
    text = text.replace(r"\subsection", "")
    text = text.replace(r"\section", "")
    text = text.replace("$", "")
    text = text.replace("{", "")
    text = text.replace("}", "")
    text = re.sub(r"\\[a-zA-Z]+", "", text)
    return text.strip()


def add_paragraph(story, text, style):
    text = clean_latex_text(text)
    if text:
        story.append(Paragraph(text, style))
        story.append(Spacer(1, 6))


def main():
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    heading_style = styles["Heading1"]
    subheading_style = styles["Heading2"]
    body_style = styles["BodyText"]
    body_style.leading = 14

    doc = SimpleDocTemplate(
        str(PDF_FILE),
        pagesize=A4,
        rightMargin=42,
        leftMargin=42,
        topMargin=42,
        bottomMargin=42,
    )

    tex = TEX_FILE.read_text(encoding="utf-8")
    lines = tex.splitlines()
    story = []
    story.append(Paragraph("Analytical Defence Notes for the 5SC28 Design Project", title_style))
    story.append(Paragraph("Group 17: Unbalanced Disk Modelling and Control", styles["Heading2"]))
    story.append(Spacer(1, 12))

    in_skip_block = False
    paragraph = []

    for line in lines:
        stripped = line.strip()

        if stripped.startswith(r"\documentclass") or stripped.startswith(r"\usepackage"):
            continue
        if stripped.startswith(r"\hypersetup") or stripped.startswith(r"\title"):
            in_skip_block = True
            continue
        if in_skip_block:
            if stripped.endswith("}") or stripped == "}":
                in_skip_block = False
            continue
        if stripped.startswith(r"\author") or stripped.startswith(r"\date"):
            continue
        if stripped in [r"\begin{document}", r"\maketitle", r"\end{document}"]:
            continue

        if stripped.startswith(r"\section"):
            add_paragraph(story, " ".join(paragraph), body_style)
            paragraph = []
            title = re.sub(r"\\section\*?\{(.+)\}", r"\1", stripped)
            story.append(Spacer(1, 10))
            story.append(Paragraph(clean_latex_text(title), heading_style))
            continue

        if stripped.startswith(r"\subsection"):
            add_paragraph(story, " ".join(paragraph), body_style)
            paragraph = []
            title = re.sub(r"\\subsection\*?\{(.+)\}", r"\1", stripped)
            story.append(Spacer(1, 6))
            story.append(Paragraph(clean_latex_text(title), subheading_style))
            continue

        if stripped.startswith(r"\includegraphics"):
            match = re.search(r"\{(.+)\}", stripped)
            if match:
                image_path = ROOT / match.group(1)
                if image_path.exists():
                    story.append(Spacer(1, 8))
                    story.append(Image(str(image_path), width=430, height=190, kind="proportional"))
                    story.append(Spacer(1, 8))
            continue

        if stripped.startswith(r"\begin") or stripped.startswith(r"\end"):
            add_paragraph(story, " ".join(paragraph), body_style)
            paragraph = []
            continue

        if stripped.startswith(r"\item"):
            add_paragraph(story, " ".join(paragraph), body_style)
            paragraph = []
            story.append(Paragraph("- " + clean_latex_text(stripped.replace(r"\item", "", 1)), body_style))
            continue

        if not stripped:
            add_paragraph(story, " ".join(paragraph), body_style)
            paragraph = []
        else:
            paragraph.append(stripped)

    add_paragraph(story, " ".join(paragraph), body_style)
    doc.build(story)
    print("Saved:", PDF_FILE)


if __name__ == "__main__":
    main()
