import os
import re
import json
import pdfplumber
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

class PDFProcessor:
    def __init__(self, input_dir="data/raw/pdfs", output_dir="data/processed"):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.processed_count = 0
        
        # Table extraction settings
        self.table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 3,
            "snap_tolerance": 3
        }

    def parse_filename_metadata(self, filename: str) -> Dict:
        """
        Extracts metadata from standardized filename:
        Format: {Bank}_{Year}_{Quarter}_{Type}.pdf
        Example: CIB_2024_Q3_Consolidated.pdf
        """
        stem = Path(filename).stem
        parts = stem.split('_')
        
        metadata = {
            "source_filename": filename,
            "bank": "Unknown",
            "year": None,
            "quarter": "Annual",
            "type": "Report",
            "processed_at": datetime.now().isoformat()
        }
        
        if len(parts) >= 4:
            metadata["bank"] = parts[0]
            metadata["year"] = int(parts[1]) if parts[1].isdigit() else parts[1]
            metadata["quarter"] = parts[2]
            metadata["type"] = parts[3]
        elif len(parts) >= 2:
            metadata["bank"] = parts[0]
            for p in parts:
                if p.isdigit() and len(p) == 4:
                    metadata["year"] = int(p)
                    
        return metadata

    def table_to_markdown(self, table: List[List]) -> str:
        """
        Converts a table array to markdown format.
        Handles None values and ensures proper alignment.
        """
        if not table or len(table) == 0:
            return ""
        
        # Clean the table data
        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip() if cell is not None else "" for cell in row]
            cleaned_table.append(cleaned_row)
        
        if len(cleaned_table) == 0:
            return ""
        
        # Calculate column widths for better formatting
        num_cols = len(cleaned_table[0])
        col_widths = [0] * num_cols
        
        for row in cleaned_table:
            for i, cell in enumerate(row):
                if i < num_cols:
                    col_widths[i] = max(col_widths[i], len(cell))
        
        # Build markdown table
        markdown_lines = []
        
        # Header row (first row)
        if len(cleaned_table) > 0:
            header = "| " + " | ".join(
                cell.ljust(col_widths[i]) for i, cell in enumerate(cleaned_table[0])
            ) + " |"
            markdown_lines.append(header)
            
            # Separator
            separator = "| " + " | ".join("-" * width for width in col_widths) + " |"
            markdown_lines.append(separator)
            
            # Data rows
            for row in cleaned_table[1:]:
                row_str = "| " + " | ".join(
                    cell.ljust(col_widths[i]) if i < len(row) else "".ljust(col_widths[i])
                    for i, cell in enumerate(row)
                ) + " |"
                markdown_lines.append(row_str)
        
        return "\n".join(markdown_lines)

    def extract_tables_from_page(self, page) -> List[Dict]:
        """
        Extracts all tables from a page and converts them to markdown.
        Returns list of table dictionaries with position info.
        """
        tables_data = []
        
        try:
            # Try with default settings first
            tables = page.extract_tables(table_settings=self.table_settings)
            
            if not tables:
                # Fallback: try without line detection
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text"
                })
            
            for idx, table in enumerate(tables):
                if table and len(table) > 1:  # At least header + 1 row
                    markdown_table = self.table_to_markdown(table)
                    
                    if markdown_table:
                        tables_data.append({
                            "table_index": idx,
                            "markdown": markdown_table,
                            "row_count": len(table),
                            "col_count": len(table[0]) if table else 0
                        })
        
        except Exception as e:
            print(f"[WARN] Table extraction error: {e}")
        
        return tables_data

    def extract_text_with_tables(self, page) -> Dict:
        """
        Extracts text and tables from a page, marking table positions.
        """
        # Extract tables first
        tables = self.extract_tables_from_page(page)
        
        # Extract text
        text = page.extract_text(layout=True)
        
        # Get table bounding boxes to identify their positions
        page_tables = page.find_tables(table_settings=self.table_settings)
        
        # Build content with tables embedded
        content_parts = []
        
        if text:
            # Split text into lines for processing
            lines = text.split('\n')
            
            # If we have tables, try to insert them at appropriate positions
            if tables and page_tables:
                # Simple approach: append tables at the end with markers
                content_parts.append(text)
                content_parts.append("\n\n<!-- TABLES EXTRACTED FROM THIS PAGE -->\n")
                
                for i, table_data in enumerate(tables):
                    content_parts.append(f"\n### Table {i + 1}\n")
                    content_parts.append(table_data["markdown"])
                    content_parts.append("\n")
            else:
                content_parts.append(text)
        
        return {
            "content": "\n".join(content_parts),
            "tables": tables,
            "has_tables": len(tables) > 0
        }

    def extract_content_from_pdf(self, pdf_path: Path) -> List[Dict]:
        """
        Extracts text and tables from PDF preserving structure.
        Returns a list of page objects.
        """
        pages_content = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for i, page in enumerate(pdf.pages):
                    # Extract with table handling
                    page_data = self.extract_text_with_tables(page)
                    
                    if page_data["content"]:
                        pages_content.append({
                            "page_number": i + 1,
                            "content": page_data["content"],
                            "has_tables": page_data["has_tables"],
                            "table_count": len(page_data["tables"]),
                            "token_count_est": len(page_data["content"].split())
                        })
                        
        except Exception as e:
            print(f"[ERROR] Failed to process PDF {pdf_path}: {e}")
            
        return pages_content

    def process_all(self):
        print(f"Scanning {self.input_dir} for PDFs...")
        output_file = self.output_dir / "extracted_financials.jsonl"
        
        with open(output_file, 'w', encoding='utf-8') as f_out:
            for root, dirs, files in os.walk(self.input_dir):
                for file in files:
                    if file.lower().endswith('.pdf'):
                        pdf_path = Path(root) / file
                        print(f"Processing: {file}")
                        
                        # 1. Metadata
                        metadata = self.parse_filename_metadata(file)
                        metadata["filepath"] = str(pdf_path)
                        
                        # 2. Content Extraction with Tables
                        pages = self.extract_content_from_pdf(pdf_path)
                        
                        if not pages:
                            print(f"[WARN] No text extracted from {file}")
                            continue

                        # 3. Save chunks
                        for page in pages:
                            doc_record = {
                                "id": f"{metadata['bank']}_{metadata['year']}_{metadata['quarter']}_{page['page_number']}",
                                "metadata": {
                                    **metadata, 
                                    "page": page['page_number'],
                                    "has_tables": page["has_tables"],
                                    "table_count": page["table_count"]
                                },
                                "content": page['content']
                            }
                            f_out.write(json.dumps(doc_record, ensure_ascii=False) + '\n')
                        
                        self.processed_count += 1
        
        print(f"\nCompleted! Processed {self.processed_count} files.")
        print(f"Output saved to {output_file}")

    def process_single_file(self, pdf_path: str):
        """
        Process a single PDF file for testing/debugging.
        """
        pdf_path = Path(pdf_path)
        print(f"Processing single file: {pdf_path.name}")
        
        metadata = self.parse_filename_metadata(pdf_path.name)
        pages = self.extract_content_from_pdf(pdf_path)
        
        print(f"\nExtracted {len(pages)} pages")
        for page in pages:
            print(f"  Page {page['page_number']}: {page['table_count']} tables, ~{page['token_count_est']} tokens")
        
        return pages


if __name__ == "__main__":
    processor = PDFProcessor()
    
    # Process all files
    processor.process_all()
    
    # Or test single file:
    # pages = processor.process_single_file("data/raw/pdfs/CIB_2024_Q3_Consolidated.pdf")
    # print(json.dumps(pages[0], indent=2, ensure_ascii=False))