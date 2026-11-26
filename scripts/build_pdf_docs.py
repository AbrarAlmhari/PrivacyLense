"""Build the PDF versions of the documentation into docs/pdf/.

Usage (from the project folder):
    python scripts/build_pdf_docs.py

How it works:
    1. Each Markdown file in docs/ is turned into HTML (library: markdown-it-py,
       which is installed together with Streamlit).
    2. The HTML is printed to PDF by a browser that is already on the computer
       (Microsoft Edge or Google Chrome, started without a window).

The Markdown files stay the single source of truth: edit them, then run this script again.
"""

import html
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

from markdown_it import MarkdownIt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
OUTPUT_DIR = DOCS_DIR / "pdf"

# One entry per PDF. A PDF with several sources gets "Part 1, Part 2, ..." headings.
DOCUMENTS = [
    {
        "output": "1_PrivacyLens_Project_Record.pdf",
        "title": "PrivacyLens Project Record",
        "subtitle": "What was built, how, why, and where the data comes from",
        "sources": ["PROJECT_RECORD.md", "DATA_SOURCES.md", "PROJECT_LOG.md", "CODE_GUIDE.md", "MODEL_CARD.md"],
    },
    {
        "output": "2_PrivacyLens_How_To_Run.pdf",
        "title": "How to Run PrivacyLens",
        "subtitle": "Installing, starting and using the app",
        "sources": ["HOW_TO_RUN.md"],
    },
    {
        "output": "3_PrivacyLens_Weak_Points_and_Improvements.pdf",
        "title": "PrivacyLens: Weak Points and Improvements",
        "subtitle": "Known limits, how we know about them, and what to do next",
        "sources": ["WEAK_POINTS_AND_IMPROVEMENTS.md"],
    },
    {
        "output": "4_PrivacyLens_Dataset_Update.pdf",
        "title": "PrivacyLens Dataset Update",
        "subtitle": "The team's ORG documents: assessment, processing, retraining and results (model version 2)",
        "sources": ["DATASET_UPDATE.md"],
    },
]

BROWSER_PATHS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]
BROWSER_COMMANDS = ["msedge", "chrome", "google-chrome", "chromium", "chromium-browser"]

STYLE = """
@page {
  size: A4;
  margin: 20mm 17mm 20mm 17mm;
  @bottom-left { content: "FOOTER_TITLE"; font: 8pt "Segoe UI", Arial, sans-serif; color: #6b7280; }
  @bottom-right { content: "Page " counter(page) " of " counter(pages); font: 8pt "Segoe UI", Arial, sans-serif; color: #6b7280; }
}
* { box-sizing: border-box; }
body { font: 10.5pt/1.5 "Segoe UI", Arial, sans-serif; color: #1f2933; margin: 0; }
.cover { border-bottom: 3px solid #1f6feb; padding-bottom: 10pt; margin-bottom: 16pt; }
.cover .course { font-size: 9pt; letter-spacing: 0.06em; text-transform: uppercase; color: #1f6feb; font-weight: 600; }
.cover .title { font-size: 25pt; font-weight: 700; line-height: 1.2; margin: 4pt 0 2pt; color: #111827; }
.cover .subtitle { font-size: 12pt; color: #4b5563; }
.cover .date { font-size: 9pt; color: #6b7280; margin-top: 6pt; }
.contents { background: #f5f8fc; border: 1px solid #dbe4f0; border-radius: 6px; padding: 10pt 14pt; margin-bottom: 8pt; }
.contents .heading { font-weight: 700; font-size: 11pt; margin-bottom: 4pt; }
.contents ol { margin: 0; padding-left: 16pt; }
.contents li { margin: 1pt 0; }
.contents ul { list-style: none; margin: 0 0 3pt; padding-left: 4pt; font-size: 9pt; columns: 2; }
.contents a { color: #1f2933; text-decoration: none; }
.part { break-before: page; }
.part.single { break-before: auto; }
.kicker { font-size: 9pt; letter-spacing: 0.06em; text-transform: uppercase; color: #1f6feb; font-weight: 600; margin-bottom: -6pt; }
h1 { font-size: 19pt; line-height: 1.25; margin: 10pt 0 8pt; color: #111827; }
h2 { font-size: 13.5pt; margin: 18pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid #d5dbe3; color: #111827; break-after: avoid; }
h3 { font-size: 11.5pt; margin: 13pt 0 4pt; color: #111827; break-after: avoid; }
p { margin: 0 0 7pt; }
ul, ol { margin: 0 0 7pt; padding-left: 18pt; }
li { margin-bottom: 2pt; }
section ul, section ol { break-inside: avoid; }
p:has(> strong:only-child) { break-after: avoid; }
a { color: #1f5fbf; text-decoration: none; overflow-wrap: anywhere; }
hr { display: none; }
code { font: 9pt Consolas, "Courier New", monospace; background: #eef1f5; padding: 0.5pt 3pt; border-radius: 3px; overflow-wrap: break-word; }
pre { background: #f3f5f8; border: 1px solid #dde2e8; border-radius: 5px; padding: 7pt 9pt; margin: 0 0 8pt; white-space: pre-wrap; overflow-wrap: anywhere; break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 8.5pt; }
table { border-collapse: collapse; width: 100%; margin: 4pt 0 10pt; font-size: 9.2pt; line-height: 1.4; }
th, td { border: 1px solid #cfd6de; padding: 4pt 6pt; text-align: left; vertical-align: top; }
th { background: #eaf0f8; font-weight: 600; }
th:empty { display: none; }
tr { break-inside: avoid; }
thead { display: table-header-group; }
blockquote { margin: 0 0 8pt; padding: 6pt 10pt; border-left: 3px solid #e0a100; background: #fff8e1; }
blockquote p { margin: 0; }
figure { margin: 8pt 0 12pt; text-align: center; break-inside: avoid; }
figure img { max-width: 100%; max-height: 150mm; border: 1px solid #d5dbe3; border-radius: 4px; }
figcaption { font-size: 8.5pt; color: #6b7280; margin-top: 3pt; }
"""


def find_browser() -> str:
    """Return the path of Edge or Chrome, or stop with a clear message."""
    for path in BROWSER_PATHS:
        if Path(path).exists():
            return path
    for command in BROWSER_COMMANDS:
        found = shutil.which(command)
        if found:
            return found
    raise SystemExit("No Edge or Chrome browser was found. Install one of them to build the PDF files.")


def slugify(text: str) -> str:
    """Turn a heading into a short id that can be used as a link target."""
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def render_markdown(markdown_text: str, prefix: str) -> tuple[str, list[tuple[int, str, str]]]:
    """Convert Markdown to HTML and collect the headings for the table of contents.

    Returns the HTML and a list of (level, heading text, id) for level 1 and 2 headings.
    """
    converter = MarkdownIt("commonmark").enable("table")
    body = converter.render(markdown_text)
    headings: list[tuple[int, str, str]] = []

    def add_id(match: re.Match) -> str:
        level, inner = int(match.group(1)), match.group(2)
        plain = html.unescape(re.sub(r"<[^>]+>", "", inner))
        anchor = f"{prefix}-{slugify(plain)}"
        headings.append((level, plain, anchor))
        return f'<h{level} id="{anchor}">{inner}</h{level}>'

    body = re.sub(r"<h([12])>(.*?)</h\1>", add_id, body)
    # A picture on its own line becomes a figure with its description as caption.
    body = re.sub(
        r'<p><img src="([^"]+)" alt="([^"]*)" /></p>',
        r'<figure><img src="\1" alt="\2"><figcaption>\2</figcaption></figure>',
        body,
    )
    return body, headings


def build_html(document: dict) -> str:
    """Build the complete HTML page for one PDF."""
    sources = document["sources"]
    several_parts = len(sources) > 1
    parts_html: list[str] = []
    contents: list[str] = []

    for number, source in enumerate(sources, start=1):
        text = (DOCS_DIR / source).read_text(encoding="utf-8")
        body, headings = render_markdown(text, prefix=f"part{number}")
        if several_parts:
            # Every part starts on a new page under a "Part 2 of 5" line.
            kicker = f'<div class="kicker">Part {number} of {len(sources)}</div>'
            parts_html.append(f'<section class="part">{kicker}{body}</section>')
        else:
            # A single document already shows its title on the cover, so the first heading is dropped.
            body = re.sub(r"<h1[^>]*>.*?</h1>", "", body, count=1)
            parts_html.append(f'<section class="part single">{body}</section>')

        part_title = next((title for level, title, _ in headings if level == 1), source)
        part_anchor = next((anchor for level, _, anchor in headings if level == 1), "")
        sections = "".join(
            f'<li><a href="#{anchor}">{html.escape(title)}</a></li>' for level, title, anchor in headings if level == 2
        )
        if several_parts:
            contents.append(f'<li><a href="#{part_anchor}"><b>{html.escape(part_title)}</b></a><ul>{sections}</ul></li>')
        else:
            contents.append(f"<ul>{sections}</ul>")

    if several_parts:
        contents_html = f'<div class="contents"><div class="heading">Contents</div><ol>{"".join(contents)}</ol></div>'
    else:
        contents_html = f'<div class="contents"><div class="heading">Contents</div>{"".join(contents)}</div>'

    today = date.today()
    cover = (
        '<div class="cover">'
        '<div class="course">ARTI 303 Programming for Artificial Intelligence &middot; Final Project</div>'
        f'<div class="title">{html.escape(document["title"])}</div>'
        f'<div class="subtitle">{html.escape(document["subtitle"])}</div>'
        f'<div class="date">Version of {today.day} {today.strftime("%B %Y")}</div>'
        "</div>"
    )
    style = STYLE.replace("FOOTER_TITLE", document["title"])
    return (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        f"<title>{html.escape(document['title'])}</title>"
        f"<base href='{DOCS_DIR.as_uri()}/'>"
        f"<style>{style}</style></head><body>{cover}{contents_html}{''.join(parts_html)}</body></html>"
    )


def print_to_pdf(browser: str, page_html: str, output_path: Path) -> None:
    """Let the browser print the HTML page to a PDF file."""
    # The browser may still be closing its files when the folder is removed; ignore that.
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as temp_dir:
        html_path = Path(temp_dir) / "page.html"
        html_path.write_text(page_html, encoding="utf-8")
        command = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--user-data-dir={Path(temp_dir) / 'profile'}",
            f"--print-to-pdf={output_path}",
            html_path.as_uri(),
        ]
        subprocess.run(command, check=True, capture_output=True, timeout=180)
    if not output_path.exists() or output_path.stat().st_size == 0:
        raise SystemExit(f"The browser did not create {output_path.name}.")


def main() -> int:
    browser = find_browser()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Browser: {browser}")
    for document in DOCUMENTS:
        output_path = OUTPUT_DIR / document["output"]
        print_to_pdf(browser, build_html(document), output_path)
        print(f"Created docs/pdf/{output_path.name} ({output_path.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
