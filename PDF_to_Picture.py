import fitz           # PyMuPDF
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────────
INPUT_DIR = Path(r"D:\PythonProject\CoreDescription\pdf_1A") # change it to your pfd file folder
OUTPUT_DIR = Path(r"D:\PythonProject\CoreDescription\picture_1A") # change it to your exported folder
DPI = 300  # adjust for higher/lower resolution
# ───────────────────────────────────────────────────────────────────────────────


OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

for pdf_path in INPUT_DIR.glob("*.pdf"):
    # Open PDF and render the first (and only) page
    doc = fitz.open(pdf_path)
    pix = doc.load_page(0).get_pixmap(dpi=DPI)
    doc.close()

    # Save as PNG with the same base filename
    out_path = OUTPUT_DIR / f"{pdf_path.stem}.png"
    pix.save(out_path)

    #print(f"Saved {out_path.name}")
