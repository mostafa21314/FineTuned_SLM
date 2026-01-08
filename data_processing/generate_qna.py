import os
import json
import time
import argparse
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from tqdm import tqdm
from google import genai
from google.genai import types
from tenacity import retry, stop_after_attempt, wait_exponential

load_dotenv()

# 1. Define the Schema (Fixes 'answer' KeyError and parsing issues)
class QnAPair(BaseModel):
    question: str
    answer: str

class QnAResponse(BaseModel):
    qna_pairs: List[QnAPair]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/clean_training_data.jsonl")
    parser.add_argument("--output", default="data/processed/qna_dataset.jsonl")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    # 2. Initialize New SDK Client
    client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    with open(args.input, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    start_index = args.start
    end_index = start_index + args.limit if args.limit else None
    lines = lines[start_index:end_index]

    # 3. Robust Retry Logic for Free Tier
    @retry(wait=wait_exponential(multiplier=2, min=15, max=60), stop=stop_after_attempt(3))
    def generate_with_retry(chunk_text):
        return client.models.generate_content(
            model="gemini-2.0-flash",
            contents=chunk_text,
            config=types.GenerateContentConfig(
                system_instruction="You are a financial analyst specialized in the 3 banks listed in the EGX30. Generate 1-3 Q&A pairs from the text. Return valid JSON.",
                response_mime_type="application/json",
                response_schema=QnAResponse, # Natively enforces the structure
                temperature=0.2,
                max_output_tokens=1024
            )
        )

    print(f"Starting generation with 2026 SDK...")
    # Open in append mode so we don't clear previous runs
    open_mode = 'a' if os.path.exists(args.output) else 'w'
    print(f"Opening {args.output} with mode '{open_mode}'...")
    with open(args.output, open_mode, encoding='utf-8') as outfile:
        success_count = 0
        for line in tqdm(lines):
            try:
                chunk_data = json.loads(line)
                text = chunk_data.get("text", "")
                
                # Call API
                response = generate_with_retry(text)
                
                # 4. Automatic Parsing (No more json.loads() needed!)
                qna_obj = response.parsed
                
                if qna_obj and qna_obj.qna_pairs:
                    for pair in qna_obj.qna_pairs:
                        entry = {
                            "chunk_id": chunk_data.get("chunk_id", "unknown"),
                            "question": pair.question, 
                            "answer": pair.answer,
                            "metadata": {
                                "bank": chunk_data.get("bank"),
                                "year": chunk_data.get("year"),
                                "quarter": chunk_data.get("quarter"),
                                "type": chunk_data.get("type"),
                                "page_number": chunk_data.get("page_number"),
                                "source_filename": chunk_data.get("source_filename")
                            }
                        }
                        outfile.write(json.dumps(entry) + "\n")
                    outfile.flush()
                    success_count += 1
                
                # 5. The "Safety Delay" for Free Tier (Reduced to 5 RPM)
                time.sleep(15) 
                
            except Exception as e:
                print(f"\nSkipping chunk due to error: {e}")
                continue

    print(f"Done. Successfully generated from {success_count} chunks.")

if __name__ == "__main__":
    main()