#!/usr/bin/env python3
"""Export analysis.ipynb to PDF with proper page breaks and formatting."""

import os
import re
import tempfile

import nbformat
from nbconvert import HTMLExporter

CUSTOM_CSS = """
<style>
@page {
    size: A4;
    margin: 2cm 1.8cm 2cm 1.8cm;
}

/* ───── Global text ───── */
body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.55;
    color: #222;
    max-width: 100%;
}

/* ───── Headings: never orphan a heading at bottom of page ───── */
h1, h2, h3, h4, h5, h6 {
    page-break-after: avoid;
    break-after: avoid;
}

h2 {
    margin-top: 1.2em;
}

/* ───── Text flows naturally across pages ───── */
.jp-RenderedHTMLCommon,
.rendered_html,
.text_cell_render,
p, ul, ol {
    page-break-inside: auto;
    break-inside: auto;
}

/* ───── Images & figures: NEVER split ───── */
.jp-RenderedImage,
.output_png,
.output_jpeg,
.output_svg,
figure,
img {
    page-break-inside: avoid;
    break-inside: avoid;
}

/* Output areas containing images: keep together */
.jp-OutputArea-output:has(img),
.output_subarea:has(img) {
    page-break-inside: avoid;
    break-inside: avoid;
}

/* ───── Tables: try to keep together but allow split if needed ───── */
table {
    border-collapse: collapse;
    width: 100%;
    font-size: 10pt;
    margin: 0.5em 0;
    page-break-inside: avoid;
    break-inside: avoid;
}

/* Compact tables with many columns — shrink to fit */
.compact-table {
    font-size: 8pt !important;
    table-layout: fixed;
}
.compact-table th, .compact-table td {
    padding: 3px 4px !important;
    overflow: hidden;
    text-overflow: ellipsis;
}

th, td {
    border: 1px solid #ccc;
    padding: 5px 8px;
    text-align: left;
}

th {
    background-color: #f5f5f5;
    font-weight: 600;
}

/* ───── Images ───── */
img {
    max-width: 100% !important;
    max-height: 85vh !important;   /* never taller than the printable area */
    width: auto !important;
    height: auto !important;
    object-fit: contain;
    display: block;
    margin: 0.5em auto;
}

/* ───── Hide code input cells ───── */
.hide-input .jp-InputArea,
.hide-input .input,
.hide-input .jp-Cell-inputWrapper {
    display: none !important;
}

/* ───── Output area cleanup ───── */
.jp-OutputArea-output pre,
.output pre {
    font-size: 9pt;
    white-space: pre-wrap;
    word-wrap: break-word;
}

/* ───── Horizontal rules ───── */
hr {
    border: none;
    border-top: 2px solid #ddd;
    margin: 1.5em 0;
}

/* ───── Remove cell prompt numbers ───── */
.jp-InputPrompt,
.jp-OutputPrompt,
.prompt {
    display: none !important;
}

/* ───── MathJax ───── */
.MathJax, .MathJax_Display {
    page-break-inside: avoid;
    break-inside: avoid;
}

/* ───── Dataframes ───── */
.dataframe {
    font-size: 9pt;
}
.dataframe th {
    background-color: #f0f0f0;
}

/* ───── Anchor links — hide the ¶ symbols ───── */
.anchor-link {
    display: none !important;
}

/* ───── Print colors ───── */
@media print {
    body {
        -webkit-print-color-adjust: exact;
        print-color-adjust: exact;
    }
    .jp-Notebook {
        padding: 0;
    }
}
</style>
"""


def _hide_setup_section(html: str) -> str:
    """Hide the Section 0: Setup heading since code is hidden and it has no
    visible content."""
    html = re.sub(
        r'<h2[^>]*id="Section-0:-Setup"[^>]*>.*?</h2>',
        "",
        html,
        flags=re.DOTALL,
    )
    return html


def export_notebook_to_pdf(
    notebook_path: str,
    output_path: str,
    include_code: bool = False,
) -> None:
    """Export notebook to PDF using HTML intermediate + playwright."""

    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    html_exporter = HTMLExporter()
    html_exporter.template_name = "classic"
    html_exporter.exclude_input = not include_code
    html_exporter.exclude_input_prompt = True
    html_exporter.exclude_output_prompt = True

    html_body, resources = html_exporter.from_notebook_node(nb)

    # Add hide-input class to body if not showing code
    if not include_code:
        html_body = html_body.replace("<body>", '<body class="hide-input">', 1)
        html_body = _hide_setup_section(html_body)

    # Inject custom CSS
    if "</head>" in html_body:
        html_body = html_body.replace("</head>", CUSTOM_CSS + "\n</head>")
    else:
        html_body = CUSTOM_CSS + html_body

    # Write HTML to temp file
    with tempfile.NamedTemporaryFile(
        suffix=".html", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(html_body)
        tmp_path = tmp.name

    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f"file://{tmp_path}", wait_until="networkidle")
            page.wait_for_timeout(2000)

            page.pdf(
                path=output_path,
                format="A4",
                margin={
                    "top": "2cm",
                    "bottom": "2cm",
                    "left": "1.8cm",
                    "right": "1.8cm",
                },
                print_background=True,
                prefer_css_page_size=False,
            )

            browser.close()

        file_size = os.path.getsize(output_path)
        print(f"Exported: {output_path}  ({file_size / 1024 / 1024:.1f} MB)")

    finally:
        os.unlink(tmp_path)


def export_linkedin_pdf() -> None:
    """Export the condensed LinkedIn notebook to PDF (no code)."""
    export_notebook_to_pdf(
        "analysis_linkedin.ipynb",
        "analysis_linkedin.pdf",
        include_code=False,
    )


if __name__ == "__main__":
    notebook = "analysis.ipynb"
    export_notebook_to_pdf(notebook, "analysis_no_code.pdf", include_code=False)
    export_notebook_to_pdf(notebook, "analysis_with_code.pdf", include_code=True)
    export_linkedin_pdf()
    print("Done.")
