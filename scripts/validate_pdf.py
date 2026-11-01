"""Usage: python scripts/validate_pdf.py FILE.pdf [FILE.musicxml [expected_bars]]  (exit 1 on problems)"""
import sys

from backend.validation.musicxml import validate_musicxml
from backend.validation.pdf import validate_pdf

if __name__ == "__main__":
    info, problems = validate_pdf(sys.argv[1])
    if len(sys.argv) > 2:
        problems += validate_musicxml(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    print(info, problems or "OK")
    sys.exit(1 if problems else 0)
