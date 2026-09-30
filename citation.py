class CitationTracker:
    def __init__(self):
        self.citations = {}
        self.counter = 1
    
    def add(self, title: str, doi: str, excerpt: str) -> int:
        """Add citation, return its index"""
        key = doi
        if key not in self.citations:
            self.citations[key] = {
                "index": self.counter,
                "title": title,
                "doi": doi,
                "excerpt": excerpt[:200]
            }
            self.counter += 1
        return self.citations[key]["index"]
    
    def format(self) -> str:
        """Return formatted citation list"""
        if not self.citations:
            return "No citations."
        
        output = "\nREFERENCES:\n"
        for doi, c in self.citations.items():
            output += f"[{c['index']}] {c['title']}\n"
            output += f"    DOI: {c['doi']}\n"
            output += f"    Excerpt: \"{c['excerpt']}\"\n\n"
        return output
    
    def reset(self):
        self.citations = {}
        self.counter = 1