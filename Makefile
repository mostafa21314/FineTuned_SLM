# EGX30 Banks Q&A Fine-Tuning Pipeline
# Run 'make help' for usage

PYTHON := python
PIP := pip

# Directories
DATA_RAW_DIR := data/raw
DATA_PROCESSED_DIR := data/processed
MODELS_DIR := models/fine_tuned

# Files
QNA_DATASET := $(DATA_PROCESSED_DIR)/qna_dataset.jsonl
CLEAN_DATA := $(DATA_PROCESSED_DIR)/clean_training_data.jsonl
CHUNKS_DATA := $(DATA_PROCESSED_DIR)/chunks.jsonl
EVAL_RESULTS := $(MODELS_DIR)/evaluation_results.json

# Model Config
BASE_MODEL := google/gemma-3-270m-it
BATCH_SIZE := 4

.PHONY: help install collect process generate train evaluate all clean

help:
	@echo "================================================================="
	@echo " EGX30 Fine-Tuning Pipeline Makefile"
	@echo "================================================================="
	@echo "Available commands:"
	@echo "  make install    - Install python dependencies and playwright"
	@echo "  make collect    - Scrap financial PDFs from CIB, ADIB, CAE"
	@echo "  make process    - Extract, chunk, clean, and filter text"
	@echo "  make generate   - Generate Q&A dataset using Gemini API"
	@echo "  make train      - Fine-tune Gemma 3 270M (LoRA)"
	@echo "  make evaluate   - Run evaluation metrics (ROUGE/BLEU/Latency)"
	@echo "  make all        - Run the entire pipeline from scratch"
	@echo "  make clean      - Remove processed data and models (Caution!)"
	@echo "================================================================="

install:
	$(PIP) install -r requirements.txt
	$(PIP) install playwright
	playwright install chromium
	@echo "[✓] Dependencies installed."

collect:
	@echo "[*] Starting Data Collection..."
	$(PYTHON) data_collection/webscraperEGX.py
	@echo "[✓] Data collection complete."

process:
	@echo "[*] Processing PDFs (Extraction)..."
	$(PYTHON) data_processing/pdf_processor.py
	@echo "[*] Chunking Text..."
	$(PYTHON) data_processing/chunk_and_clean.py
	@echo "[*] Filtering and Validating..."
	$(PYTHON) data_processing/filter_and_validate.py
	@echo "[✓] Data processing complete. Ready for Q&A generation."

generate:
	@echo "[*] Generating Q&A Dataset (using Gemini API)..."
	# Adjust limit as needed, default 100 for demo
	$(PYTHON) data_processing/generate_qna.py --start 0 --limit 1000
	@echo "[✓] Dataset generation complete: $(QNA_DATASET)"

train:
	@echo "[*] Starting Fine-Tuning ($(BASE_MODEL))..."
	$(PYTHON) models/fine_tune.py \
		--model_name "$(BASE_MODEL)" \
		--output_dir "$(MODELS_DIR)" \
		--epochs 3 \
		--batch_size $(BATCH_SIZE) \
		--disable_quantization \
		--input "$(QNA_DATASET)" \
		--chunks "$(CHUNKS_DATA)"
	@echo "[✓] Training complete. Adapters saved to $(MODELS_DIR)"

evaluate:
	@echo "[*] Running Evaluation..."
	$(PYTHON) models/evaluate_model.py \
		--base_model_id "$(BASE_MODEL)" \
		--adapter_path "$(MODELS_DIR)" \
		--chunks_file "$(CHUNKS_DATA)" \
		--batch_size 8 \
		--limit 50
	@echo "[✓] Evaluation complete. Results in $(EVAL_RESULTS)"

all: install collect process generate train evaluate
	@echo "[🎉] Full pipeline completed successfully!"

clean:
	@echo "[!] Cleaning up processed data and models..."
	rm -rf $(DATA_PROCESSED_DIR)/*.jsonl
	rm -rf $(MODELS_DIR)
	@echo "[✓] Cleanup complete."
