"""PDF checks: valid header, page count > 0, first page rasterizes to a non-blank image."""
import re
import subprocess


def validate_pdf(path):
    """Return (info, problems)."""
    problems = []
    with open(path, "rb") as f:
        if f.read(4) != b"%PDF":
            return {}, ["missing %PDF header"]
    meta = subprocess.run(["pdfinfo", str(path)], check=False, capture_output=True, text=True)
    m = re.search(r"Pages:\s+(\d+)", meta.stdout)
    pages = int(m.group(1)) if m else 0
    if pages < 1:
        problems.append("pdfinfo reports no pages")
    ink = None
    if pages:
        pgm = subprocess.run(["pdftoppm", "-r", "40", "-gray", "-f", "1", "-l", "1", str(path)],
                             check=False, capture_output=True).stdout
        hdr = re.match(rb"P5\s+(\d+)\s+(\d+)\s+(\d+)\s", pgm)
        if hdr:
            px = pgm[hdr.end():]
            ink = sum(b < 128 for b in px) / max(len(px), 1)
            if ink < 1e-4:
                problems.append("first page is blank")
        else:
            problems.append("could not rasterize page 1")
    return {"pages": pages, "ink_fraction_p1": ink}, problems
