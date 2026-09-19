import os
from pathlib import Path
from pypdf import PdfReader

BASE_DIR = Path(__file__).resolve().parent
PAPERS_DIR = BASE_DIR / "papers"
EXTRACTED_DIR = BASE_DIR / "extracted_text"
EXTRACTED_DIR.mkdir(exist_ok=True)

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF"""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def process_papers():
    """Extract text from all PDFs"""
    pdfs = list(PAPERS_DIR.glob("*.pdf"))
    
    for i, pdf_path in enumerate(pdfs):
        pmid = pdf_path.stem
        print(f"\n[{i+1}/{len(pdfs)}] {pmid}")
        
        # Skip if already extracted
        if os.path.exists(f"{EXTRACTED_DIR}/{pmid}_text.txt"):
            print("  Already extracted")
            continue
        
        text = extract_text_from_pdf(pdf_path)
        if text:
            with open(f"{EXTRACTED_DIR}/{pmid}_text.txt", "w", encoding="utf-8") as f:
                f.write(text)
            print(f"  ✓ Extracted {len(text)} chars")
        else:
            print(f"  ✗ Failed")

if __name__ == "__main__":
    process_papers()
    print("\n✓ Done")