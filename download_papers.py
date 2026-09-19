import csv
import json
import os
import re
import time
from pathlib import Path

import requests

PAPERS_DIR = Path("papers")
METADATA_DIR = Path("metadata")
PAPERS_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)


def is_pdf_response(resp):
    """Return True for actual PDF payloads or valid PDF endpoints."""
    content_type = (resp.headers.get("Content-Type") or "").lower()
    content_disposition = (resp.headers.get("Content-Disposition") or "").lower()
    final_url = (resp.url or "").lower()

    if resp.content.startswith(b"%PDF"):
        return True
    if "application/pdf" in content_type or "x-pdf" in content_type:
        return True
    if "pdf" in final_url or ".pdf" in final_url:
        return True
    if "pdf" in content_disposition:
        return True
    return False


def save_pdf_response(resp, pmid, source):
    """Persist response content only if it is a genuine PDF."""
    if not is_pdf_response(resp):
        return False

    filename = f"{PAPERS_DIR}/{pmid}.pdf"
    with open(filename, "wb") as f:
        f.write(resp.content)
    print(f"✓ Downloaded via {source}: {pmid}")
    return True


def download_pdf_from_pmc(pmcid, pmid):
    """Download PDF from PMC"""
    url = f"https://www.ncbi.nlm.nih.gov/pmc/articles/{pmcid}/pdf/"
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            # Validate PDF header
            if resp.content.startswith(b"%PDF"):
                filename = f"{PAPERS_DIR}/{pmid}.pdf"
                with open(filename, "wb") as f:
                    f.write(resp.content)
                print(f"✓ Downloaded from PMC: {pmid}")
                return True
            else:
                print(f"✗ Invalid PDF (not %PDF header)")
                return False
    except Exception as e:
        print(f"✗ PMC failed {pmid}: {e}")
    return False


def download_pdf_from_doi(doi, pmid):
    """Try to get PDF via DOI, allowing publisher redirect pages and .pdf links."""
    if not doi or doi.strip() == "":
        return False

    url = f"https://doi.org/{doi}"
    try:
        resp = requests.get(
            url,
            allow_redirects=True,
            timeout=20,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if resp.status_code == 200:
            if save_pdf_response(resp, pmid, "DOI"):
                return True

            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "text/html" in content_type:
                match = re.search(r"https?://[^\"'\s>]+\.pdf(?:\?[^\"'\s>]*)?", resp.text, re.I)
                if match:
                    pdf_url = match.group(0)
                    pdf_resp = requests.get(pdf_url, timeout=20, allow_redirects=True, headers={"User-Agent": "Mozilla/5.0"})
                    if pdf_resp.status_code == 200 and save_pdf_response(pdf_resp, pmid, "DOI PDF link"):
                        return True
    except Exception as e:
        print(f"✗ DOI failed {pmid}: {e}")
    return False

def save_metadata(pmid, title, authors, doi, pub_year):
    """Save metadata as JSON"""
    metadata = {
        "pmid": pmid,
        "title": title,
        "authors": authors,
        "doi": doi,
        "publication_year": pub_year
    }
    filename = f"{METADATA_DIR}/{pmid}_metadata.json"
    with open(filename, "w") as f:
        json.dump(metadata, f, indent=2)

def process_csv(csv_file):
    """Read CSV and download papers"""
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    processed = 0
    for i, row in enumerate(rows):
        pmid = row.get("PMID", "").strip()
        title = row.get("Title", "").strip()
        authors = row.get("Authors", "").strip()
        doi = row.get("DOI", "").strip()
        pmcid = row.get("PMCID", "").strip()
        pub_year = row.get("Publication Year", "").strip()
        
        if not pmid:
            continue
        
        processed += 1
        print(f"\n[{processed}] PMID: {pmid}")
        
        # Check if already downloaded
        if os.path.exists(f"{PAPERS_DIR}/{pmid}.pdf"):
            print(f"  Already exists")
            continue
        
        # Try PMC first
        if pmcid and download_pdf_from_pmc(pmcid, pmid):
            save_metadata(pmid, title, authors, doi, pub_year)
            time.sleep(1)
            continue
        
        # Fallback to DOI
        if download_pdf_from_doi(doi, pmid):
            save_metadata(pmid, title, authors, doi, pub_year)
            time.sleep(1)
            continue
        
        print(f"  ✗ Failed")
        time.sleep(1)

    print(f"\n✓ Done: {processed} papers processed")

if __name__ == "__main__":
    process_csv("papers_list.csv")