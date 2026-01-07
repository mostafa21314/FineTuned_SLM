import json
import re
from pathlib import Path
from typing import List, Dict

class TextChunker:
    def __init__(self, input_file="data/processed/extracted_financials.jsonl", output_file="data/processed/chunks.jsonl"):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.target_chunk_size = 2500  # Approx 500-800 tokens
        self.min_page_length = 200

    def clean_text(self, text: str) -> str:
        if not text:
            return ""
        # normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def chunk_text(self, text: str) -> List[str]:
        """
        Splits text into chunks of approximately target_chunk_size characters.
        Tries to respect sentence boundaries.
        """
        chunks = []
        current_chunk = []
        current_length = 0
        
        # Split by sentence-ending punctuation followed by space
        # This is a simple heuristic; sophisticated (nltk/spacy) splitters could be used later
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        for sentence in sentences:
            sentence_len = len(sentence)
            
            # If adding this sentence exceeds target substantially (10% buffer), flush current
            if current_chunk and (current_length + sentence_len > self.target_chunk_size * 1.1):
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0
            
            current_chunk.append(sentence)
            current_length += sentence_len
            
            # If purely this sentence is massive (e.g. table data parsed as one string), split it forcefully? 
            # For now, we allow it to slightly exceed or be its own chunk.
            
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks

    def process(self):
        print(f"Reading from {self.input_file}...")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        stats = {
            "processed_pages": 0,
            "skipped_pages": 0,
            "total_chunks": 0
        }
        
        with open(self.input_file, 'r', encoding='utf-8') as f_in, \
             open(self.output_file, 'w', encoding='utf-8') as f_out:
            
            for line in f_in:
                try:
                    data = json.loads(line)
                    raw_text = data.get('content', '')
                    metadata = data.get('metadata', {})
                    
                    # 1. Clean
                    cleaned_text = self.clean_text(raw_text)
                    
                    # 2. Filter
                    if len(cleaned_text) < self.min_page_length:
                        stats["skipped_pages"] += 1
                        continue
                        
                    # 3. Chunk
                    text_chunks = self.chunk_text(cleaned_text)
                    
                    # 4. Save
                    for i, chunk_text in enumerate(text_chunks):
                        chunk_id = f"{metadata.get('bank')}_{metadata.get('year')}_{metadata.get('quarter')}_{metadata.get('type')}_p{metadata.get('page')}_c{i}"
                        
                        record = {
                            "chunk_id": chunk_id,
                            "bank": metadata.get('bank'),
                            "year": metadata.get('year'),
                            "quarter": metadata.get('quarter'),
                            "type": metadata.get('type'),
                            "page_number": metadata.get('page'),
                            "text": chunk_text,
                            "source_filename": metadata.get('source_filename')
                        }
                        
                        f_out.write(json.dumps(record, ensure_ascii=False) + '\n')
                        stats["total_chunks"] += 1
                        
                    stats["processed_pages"] += 1
                    
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON line")
                    continue
                    
        print(f"Chunking complete.")
        print(f"Pages processed: {stats['processed_pages']}")
        print(f"Pages skipped (<{self.min_page_length} chars): {stats['skipped_pages']}")
        print(f"Total chunks created: {stats['total_chunks']}")
        print(f"Output saved to {self.output_file}")

if __name__ == "__main__":
    chunker = TextChunker()
    chunker.process()
