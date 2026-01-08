# EGX30 Banks Q&A Fine-Tuning Pipeline

This project implements a complete pipeline for fine-tuning **Google Gemma 3 270M** on a custom financial Question-Answering dataset derived from EGX30 bank reports (CIB, ADIB, CAE).

The pipeline covers the entire data lifecycle:
1.  **Data Collection**: Web scraping financial statements from official bank websites.
2.  **Processing**: Extracting text and tables from PDFs.
3.  **Cleaning & Chunking**: preparing text for RAG/Training.
4.  **Dataset Generation**: Creating Q&A pairs using Gemini API.
5.  **Fine-Tuning**: Training the model with LoRA.
6.  **Evaluation**: Benchmarking performance.

---

## 🛠️ Prerequisites

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: For CIB scraping, you also need Playwright:*
    ```bash
    pip install playwright
    playwright install
    ```

2.  **API Keys & Setup**:
    *   **Google Gemini API**: Create a `.env` file:
        ```bash
        GOOGLE_API_KEY=your_api_key_here
        ```
    *   **Hugging Face** (for model access):
        ```bash
        huggingface-cli login
        ```
    *   **Weights & Biases** (for logging):
        ```bash
        wandb login
        ```

---

## 🚀 Running the Pipeline

### Phase 1: Data Collection & Processing

#### 1. Web Scraping
Download financial statements (PDFs) for CIB, ADIB, and CAE.
```bash
python data_collection/webscraperEGX.py
```
*   **Output**: `data/raw/pdfs/{Bank}/{Year}/...`

#### 2. PDF Processing (Extraction)
Extract text and tables from the downloaded PDFs, preserving structure.
```bash
python data_processing/pdf_processor.py
```
*   **Output**: `data/processed/extracted_financials.jsonl`

#### 3. Chunking
Split the extracted text into manageable chunks (approx. 500-800 tokens).
```bash
python data_processing/chunk_and_clean.py
```
*   **Output**: `data/processed/chunks.jsonl`

#### 4. Filtering (Quality Control)
Remove boilerplate strings (e.g., "Table of Contents"), short chunks, and low-information text. Deduplicate content.
```bash
python data_processing/filter_and_validate.py
```
*   **Output**: `data/processed/clean_training_data.jsonl`

---

### Phase 2: Dataset Generation

Generate high-quality Q&A pairs from the *filtered* chunks using the Gemini API. This allows the model to learn from specific financial contexts.

```bash
# Process clean chunks starting from index 0
python data_processing/generate_qna.py --start 0 --limit 100
```
*   **Input**: `data/processed/clean_training_data.jsonl`
*   **Output**: `data/processed/qna_dataset.jsonl`

---

### Phase 3: Fine-Tuning

Fine-tune **Gemma 3 270M** using LoRA.
*   **Important**: We use `data/processed/chunks.jsonl` (the full knowledge base) for context augmentation during training, and `qna_dataset.jsonl` for the actual QA pairs.

```bash
python models/fine_tune.py \
    --model_name "google/gemma-3-270m-it" \
    --output_dir "models/fine_tuned" \
    --epochs 3 \
    --batch_size 4 \
    --disable_quantization \
    --input "data/processed/qna_dataset.jsonl" \
    --chunks "data/processed/chunks.jsonl"
```

**Why `--disable_quantization`?**
For small models like the 270M parameter version, 4-bit quantization can degrade performance significantly. We recommend running in native `bfloat16` or `float16`.

---

### Phase 4: Evaluation

Benchmark the fine-tuned model against the base model using ROUGE, BLEU, and throughput metrics.

```bash
python models/evaluate_model.py \
    --base_model_id "google/gemma-3-270m-it" \
    --adapter_path "models/fine_tuned" \
    --chunks_file "data/processed/chunks.jsonl" \
    --batch_size 8 \
    --limit 50       
```
Note that you can change the limit to evaluate more samples or remove it completly to evaluate on the entire test set.
---

### Phase 5: Inference (Chat)

Interact with your fine-tuned model.

```bash
python models/inference.py \
    --model_id "google/gemma-3-270m-it" \
    --adapter_path "models/fine_tuned" \
    --disable_quantization
```

## 📂 Directory Structure

```
├── data/
│   ├── raw/pdfs/              # Downloaded PDFs
│   ├── processed/
│   │   ├── extracted_financials.jsonl
│   │   ├── chunks.jsonl       # All Text Chunks (KB)
│   │   ├── clean_training_data.jsonl # High-quality subset
│   │   └── qna_dataset.jsonl  # Generated Training Data
├── data_collection/
│   ├── webscraperEGX.py       # EGX30 Scraper (CIB, ADIB, CAE)
├── data_processing/
│   ├── pdf_processor.py       # PDF -> Text/Tables
│   ├── chunk_and_clean.py     # Text -> Chunks
│   ├── filter_and_validate.py # Quality Filter
│   ├── generate_qna.py        # Gemini Data Gen
├── models/
│   ├── fine_tune.py           # LoRA Training
│   ├── evaluate_model.py      # Benchmarking
│   ├── inference.py           # Chat
```