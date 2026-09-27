"""Small, purpose-built Markdown -> HTML converter for the lecture-package PDFs.
Handles exactly the subset of Markdown used in these notes: headers, bold/italic,
inline code, fenced code blocks, tables, hr, ordered/unordered/checkbox lists,
blockquotes, links, and $$...$$ display formulas (rendered as a styled block, no MathJax).
"""
import html
import re
import sys
from pathlib import Path


def inline(s: str) -> str:
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)([^*]+?)\*(?!\*)", r"<em>\1</em>", s)
    s = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', s)
    return s


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]+>", "", s).lower()).strip("-")


def table_html(rows: list[str]) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, _, *body = cells
    out = ["<table>", "<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in header) + "</tr></thead>", "<tbody>"]
    for r in body:
        out.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
    out.append("</tbody></table>")
    return "\n".join(out)


def convert(md: str) -> tuple[str, list[tuple[int, str, str]]]:
    lines = md.split("\n")
    out, toc, i, in_code, in_list = [], [], 0, False, None
    while i < len(lines):
        line = lines[i]
        if line.startswith("```"):
            fence, buf = line[3:].strip(), []
            i += 1
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(lines[i]); i += 1
            i += 1
            out.append(f'<pre class="lang-{fence or "text"}"><code>{html.escape(chr(10).join(buf))}</code></pre>')
            continue
        if line.startswith("$$"):
            buf = []
            i += 1
            while i < len(lines) and not lines[i].startswith("$$"):
                buf.append(lines[i]); i += 1
            i += 1
            out.append(f'<div class="formula">{html.escape(chr(10).join(buf).strip())}</div>')
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            anchor = slug(text)
            out.append(f'<h{level} id="{anchor}">{inline(text)}</h{level}>')
            toc.append((level, text, anchor))
            i += 1
            continue
        if line.strip() == "---":
            out.append("<hr/>"); i += 1; continue
        if line.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|?[\s:-]+\|", lines[i + 1]):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i]); i += 1
            out.append(table_html(rows))
            continue
        if re.match(r"^>\s?", line):
            buf = []
            while i < len(lines) and re.match(r"^>\s?", lines[i]):
                buf.append(re.sub(r"^>\s?", "", lines[i])); i += 1
            out.append(f'<blockquote>{"<br/>".join(inline(b) for b in buf)}</blockquote>')
            continue
        if re.match(r"^\s*[-*]\s+\[[ xX]\]\s+", line):
            if in_list != "cb":
                if in_list:
                    out.append(f"</{in_list}>")
                out.append('<ul class="checklist">'); in_list = "ul"
            checked = "x" in re.match(r"^\s*[-*]\s+\[([ xX])\]", line).group(1).lower()
            text = re.sub(r"^\s*[-*]\s+\[[ xX]\]\s+", "", line)
            box = "☑" if checked else "☐"
            out.append(f"<li>{box} {inline(text)}</li>")
            i += 1
            continue
        if re.match(r"^\s*[-*]\s+", line):
            if in_list != "ul":
                if in_list:
                    out.append(f"</{in_list}>")
                out.append("<ul>"); in_list = "ul"
            item_text = re.sub(r"^\s*[-*]\s+", "", line)
            out.append(f"<li>{inline(item_text)}</li>")
            i += 1
            continue
        if re.match(r"^\s*\d+\.\s+", line):
            if in_list != "ol":
                if in_list:
                    out.append(f"</{in_list}>")
                out.append("<ol>"); in_list = "ol"
            item_text = re.sub(r"^\s*\d+\.\s+", "", line)
            out.append(f"<li>{inline(item_text)}</li>")
            i += 1
            continue
        if in_list and line.strip() == "":
            out.append(f"</{in_list}>"); in_list = None
        if line.strip() == "":
            i += 1; continue
        out.append(f"<p>{inline(line.strip())}</p>")
        i += 1
    if in_list:
        out.append(f"</{in_list}>")
    return "\n".join(out), toc


CSS = """
@page { size: A4; margin: 20mm 16mm 18mm 16mm;
  @bottom-center { content: counter(page) " / " counter(pages); font-size: 9px; color: #8a8984; } }
* { box-sizing: border-box; }
body { font-family: 'Georgia', 'Times New Roman', serif; color: #1c1c1a; line-height: 1.5; font-size: 12.5px; }
h1, h2, h3, h4 { font-family: 'Helvetica Neue', Arial, sans-serif; color: #16325c; page-break-after: avoid; }
h1 { font-size: 22px; border-bottom: 2.5px solid #2a78d6; padding-bottom: 6px; margin-top: 0; }
h2 { font-size: 17px; border-bottom: 1px solid #c9c7c0; padding-bottom: 3px; margin-top: 26px; }
h3 { font-size: 14px; color: #2a5c9a; margin-top: 18px; }
h4 { font-size: 12.5px; color: #4a3aa7; margin-top: 14px; }
p { margin: 6px 0; text-align: justify; }
code { background: #f0efe9; border-radius: 3px; padding: 1px 4px; font-family: 'SFMono-Regular', Menlo, Consolas, monospace; font-size: 11px; }
pre { background: #f5f4f0; border: 1px solid #ddd; border-radius: 5px; padding: 10px; overflow-x: auto; page-break-inside: avoid; }
pre code { background: none; padding: 0; font-size: 10.5px; line-height: 1.4; }
.formula { background: #eef3fa; border-left: 3px solid #2a78d6; padding: 10px 14px; margin: 10px 0; font-family: 'Cambria Math', Georgia, serif; font-size: 13px; text-align: center; }
table { border-collapse: collapse; width: 100%; margin: 10px 0 16px; page-break-inside: avoid; font-size: 11px; }
th, td { border: 1px solid #cfcdc6; padding: 5px 7px; text-align: left; vertical-align: top; }
th { background: #16325c; color: #fff; font-family: Arial, sans-serif; font-size: 10.5px; }
tr:nth-child(even) td { background: #f7f6f2; }
blockquote { border-left: 3px solid #eda100; background: #fdf6e8; margin: 10px 0; padding: 8px 14px; font-style: italic; color: #5a4b1e; }
ul, ol { margin: 6px 0 6px 22px; padding: 0; }
li { margin: 3px 0; }
ul.checklist { list-style: none; margin-left: 4px; }
hr { border: none; border-top: 1px solid #ddd; margin: 18px 0; }
a { color: #2a78d6; }
.cover { page-break-after: always; display: flex; flex-direction: column; justify-content: center; align-items: center;
  height: 250mm; text-align: center; }
.cover .eyebrow { font-family: Arial, sans-serif; letter-spacing: 3px; text-transform: uppercase; color: #2a78d6; font-size: 12px; margin-bottom: 18px; }
.cover h1 { border: none; font-size: 30px; max-width: 480px; }
.cover .subtitle { font-family: Arial, sans-serif; font-size: 15px; color: #52514e; margin-top: 10px; }
.cover .meta { margin-top: 60px; font-family: Arial, sans-serif; font-size: 11px; color: #8a8984; line-height: 1.8; }
.toc { page-break-after: always; }
.toc ul { list-style: none; margin-left: 0; }
.toc li { margin: 4px 0; }
.toc a { text-decoration: none; color: #1c1c1a; }
.toc .lvl-1 { font-weight: bold; margin-top: 14px; font-size: 13px; }
.toc .lvl-2 { margin-left: 18px; font-size: 11.5px; color: #4a4a48; }
.section-break { page-break-before: always; }
.badge { display: inline-block; font-family: Arial, sans-serif; font-size: 10px; letter-spacing: 1px; text-transform: uppercase;
  background: #eb6834; color: #fff; padding: 3px 10px; border-radius: 10px; margin-bottom: 10px; }
"""


def main():
    out_html = Path(sys.argv[1])
    cover_title, cover_subtitle, cover_meta = sys.argv[2], sys.argv[3], sys.argv[4]
    parts = sys.argv[5:]  # badge:md_path pairs
    body_sections = []
    full_toc = []
    for spec in parts:
        badge, path = spec.split(":", 1)
        html_body, toc = convert(Path(path).read_text())
        body_sections.append((badge, html_body))
        full_toc.extend(toc)

    toc_html = ['<div class="toc"><h1>Table of Contents</h1><ul>']
    for level, text, anchor in full_toc:
        if level <= 2:
            toc_html.append(f'<li class="lvl-{level}"><a href="#{anchor}">{inline(text)}</a></li>')
    toc_html.append("</ul></div>")

    sections_html = []
    for idx, (badge, body) in enumerate(body_sections):
        cls = "section-break" if idx > 0 else ""
        sections_html.append(f'<div class="{cls}"><span class="badge">{badge}</span>{body}</div>')

    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Lecture 1 Package</title><style>{CSS}</style></head><body>
<div class="cover">
  <div class="eyebrow">Master in Financial Analysis and Algorithmic Trading</div>
  <h1>{cover_title}</h1>
  <div class="subtitle">{cover_subtitle}</div>
  <div class="meta">{cover_meta}</div>
</div>
{''.join(toc_html)}
{''.join(sections_html)}
</body></html>"""
    out_html.write_text(doc)
    print(f"wrote {out_html} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
