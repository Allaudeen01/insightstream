#!/usr/bin/env python3
"""Analyze PDF page count and content."""
import sys
from pypdf import PdfReader

pdf_path = sys.argv[1] if len(sys.argv) > 1 else "InsightStream-Report-NO-BLANK-PAGE.pdf"

try:
    reader = PdfReader(pdf_path)
    total_pages = len(reader.pages)
    
    print(f"\n{'='*60}")
    print(f"PDF Analysis: {pdf_path}")
    print(f"{'='*60}")
    print(f"Total pages: {total_pages}\n")
    
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text()
        char_count = len(text)
        line_count = len(text.split('\n'))
        
        # Check if page is likely blank (very few characters)
        status = "❌ BLANK" if char_count < 50 else "✅"
        
        print(f"Page {i:2d}: {status:8s} | {char_count:5d} chars | {line_count:3d} lines")
        
        # Show first 100 chars for context
        if char_count > 0:
            preview = text[:100].replace('\n', ' ')
            print(f"         Preview: {preview}...")
        print()
    
    print(f"{'='*60}")
    print(f"Summary: {total_pages} pages total")
    blank_pages = sum(1 for page in reader.pages if len(page.extract_text()) < 50)
    if blank_pages > 0:
        print(f"⚠️  WARNING: {blank_pages} blank page(s) detected!")
    else:
        print("✅ No blank pages detected!")
    print(f"{'='*60}\n")
    
except Exception as e:
    print(f"Error analyzing PDF: {e}")
    sys.exit(1)
