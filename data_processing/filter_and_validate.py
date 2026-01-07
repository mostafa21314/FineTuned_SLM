import json
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class QualityFilter:
    def __init__(self, input_file="data/processed/chunks.jsonl", output_file="data/processed/clean_training_data.jsonl"):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        # Thresholds
        self.min_tokens = 100
        self.min_number_count = 8
        self.similarity_threshold = 0.90
        
        # Boilerplate patterns
        self.boilerplate_patterns = [
            re.compile(r'^table of contents', re.IGNORECASE),
            re.compile(r'forward[- ]looking statements', re.IGNORECASE),
            re.compile(r'^disclaimer', re.IGNORECASE),
            re.compile(r'auditor\'s report', re.IGNORECASE),
            re.compile(r'^page \d+$', re.IGNORECASE | re.MULTILINE)
        ]

    def is_boilerplate(self, text: str) -> bool:
        # Check specific patterns
        for pattern in self.boilerplate_patterns:
            if pattern.search(text[:500]): # Check start of text usually
                return True
        return False

    def process(self):
        print(f"Loading chunks from {self.input_file}...")
        
        chunks = []
        valid_indices = []
        
        if not self.input_file.exists():
            print(f"[ERROR] Input file {self.input_file} not found.")
            return

        # 1. Load and Basic Filtering
        stats = {"total": 0, "short": 0, "boilerplate": 0, "low_numeric": 0, "kept_pre_dedup": 0}
        
        with open(self.input_file, 'r', encoding='utf-8') as f:
            for line in f:
                stats["total"] += 1
                try:
                    data = json.loads(line)
                    text = data.get('text', '')
                    
                    # Length Filter (approx token count)
                    if len(text.split()) < self.min_tokens:
                        stats["short"] += 1
                        continue
                        
                    # Numeric Density Filter (Ensure financial relevance)
                    # Count sequences of digits
                    numbers = re.findall(r'\d+', text)
                    if len(numbers) < self.min_number_count:
                        stats["low_numeric"] += 1
                        continue

                    # Boilerplate Filter
                    if self.is_boilerplate(text):
                        stats["boilerplate"] += 1
                        continue
                        
                    chunks.append(data)
                except json.JSONDecodeError:
                    continue

        stats["kept_pre_dedup"] = len(chunks)
        print(f"Initial Filter Stats: {stats}")
        
        if not chunks:
            print("No chunks remained after filtering.")
            return

        # 2. Deduplication (Cosine Similarity)
        print("Running deduplication...")
        texts = [c['text'] for c in chunks]
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        # We can do this efficiently by checking similarity against *accepted* set
        # But for full n*n check on large dataset, this can be slow.
        # Strategy: Greedy approach. Keep first, drop subsequent similar ones.
        
        keep_indices = []
        # Calculate similarity matrix
        # Note: For very large datasets, use batch processing or LSH.
        # Assuming extracted_financials is reasonable size (e.g. < 10k chunks)
        
        # Optimization: matrix multiplication
        sim_matrix = cosine_similarity(tfidf_matrix)
        
        # Mask lower triangle to avoid self-comparison and double counting
        # actually, we just iterate.
        
        dropped_indices = set()
        for i in range(len(chunks)):
            if i in dropped_indices:
                continue
                
            keep_indices.append(i)
            
            # Find duplicates for this chunk
            # distinct chunks j > i
            similarities = sim_matrix[i]
            
            # identify indices > i where similarity > threshold
            dupes = np.where(similarities[i+1:] > self.similarity_threshold)[0]
            
            # adjust index because we sliced [i+1:]
            dupes_indices = dupes + (i + 1)
            
            for d_idx in dupes_indices:
                dropped_indices.add(d_idx)

        print(f"Deduplication finished. Dropped {len(dropped_indices)} duplicates.")
        
        # 3. Save Output
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_file, 'w', encoding='utf-8') as f_out:
            for idx in keep_indices:
                f_out.write(json.dumps(chunks[idx], ensure_ascii=False) + '\n')
                
        print(f"Final dataset saved to {self.output_file}")
        print(f"Total chunks: {len(keep_indices)}")

if __name__ == "__main__":
    filter_process = QualityFilter()
    filter_process.process()
