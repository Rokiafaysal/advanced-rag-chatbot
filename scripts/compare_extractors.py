import sys 

import pymupdf
from pypdf import PdfReader

path =sys.argv[1]
pages=[int(p) for p in sys.argv[2:]]

pypdf_doc=PdfReader(path)
mupdf_doc=pymupdf.open(path)
NO_LIGATURES = pymupdf.TEXTFLAGS_TEXT & ~pymupdf.TEXT_PRESERVE_LIGATURES

for n in pages:
    page = mupdf_doc[n - 1]
    print(f"\n{'=' * 25} page {n} {'=' * 25}")
    print("--- pypdf ---")
    print((pypdf_doc.pages[n - 1].extract_text() or "")[:400])
    print("\n--- PyMuPDF sort=True ---")
    print(page.get_text("text", sort=True)[:400])
    print("\n--- PyMuPDF sort=False ---")
    print(page.get_text("text")[:400])
    print("\n--- PyMuPDF sort=False, no ligatures ---")
    print(page.get_text("text", flags=NO_LIGATURES)[:400])