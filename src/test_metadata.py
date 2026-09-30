"""测试元数据自动提取"""
import sys
sys.path.insert(0, "F:/AI_Project/AI-Paper-Assistant/src")
from pathlib import Path
from pdf_parser import StructuredPDFParser

PAPERS_DIR = Path("F:/AI_Project/AI-Paper-Assistant/data/papers")
pdfs = list(PAPERS_DIR.glob("*.pdf"))

for pdf in pdfs[:3]:
    print(f"\n{'='*50}")
    print(f"PDF: {pdf.name}")
    parser = StructuredPDFParser(pdf)
    print(f"标题: {parser.metadata['title'][:80]}")
    print(f"作者: {parser.metadata['authors']}")
    print(f"年份: {parser.metadata['year']}")
