"""SEC EDGAR Filing Scraper - 3-phase approach: Discovery, Mapping, Download"""

import json
import csv
import time
import re
import requests
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from urllib.parse import urljoin
import yaml
from loguru import logger
from bs4 import BeautifulSoup


class SECFilingScraper:
    
    def __init__(self, config_path: str = "config.yaml", output_dir: str = "data/raw/sec_filings"):
        self.config = self._load_config(config_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.min_request_interval = 0.1
        
        self.headers = {
            "User-Agent": "Mostafa mostafa21314@aucegypt.edu",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "keep-alive",
        }
        
        self.session = requests.Session()
        self.session.headers.update(self.headers)
        self.ticker_to_cik: Dict[str, str] = {}
        
    def _load_config(self, config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _rate_limit(self):
        time.sleep(self.min_request_interval)
    
    def _make_request(self, url: str, **kwargs) -> Optional[requests.Response]:
        self._rate_limit()
        try:
            response = self.session.get(url, timeout=30, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            return None
    
    def get_tickers(self) -> List[str]:
        tickers = self.config.get('data_sources', {}).get('sec_edgar', {}).get('tickers', [])
        logger.info(f"Target tickers: {tickers}")
        return tickers
    
    def map_ticker_to_cik(self, ticker: str) -> Optional[str]:
        if ticker in self.ticker_to_cik:
            return self.ticker_to_cik[ticker]
        
        url = "https://www.sec.gov/files/company_tickers.json"
        response = self._make_request(url)
        
        if not response:
            return None
        
        try:
            data = response.json()
            for entry in data.values():
                if entry.get('ticker') == ticker:
                    cik = str(entry.get('cik_str', '')).zfill(10)
                    self.ticker_to_cik[ticker] = cik
                    logger.info(f"Mapped {ticker} -> CIK {cik}")
                    return cik
        except Exception as e:
            logger.error(f"Error parsing ticker mapping: {e}")
        
        return None
    
    def map_all_tickers_to_ciks(self) -> Dict[str, str]:
        tickers = self.get_tickers()
        cik_map = {}
        
        for ticker in tickers:
            cik = self.map_ticker_to_cik(ticker)
            if cik:
                cik_map[ticker] = cik
        
        logger.info(f"Mapped {len(cik_map)} tickers to CIKs")
        return cik_map
    
    def parse_master_index(self, content: str) -> List[Dict]:
        filings = []
        lines = content.split('\n')
        data_started = False
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith('Description:') or line.startswith('Last Data Received:'):
                continue
            
            if '---' in line:
                data_started = True
                continue
            
            if not data_started:
                continue
            
            parts = line.split('|')
            if len(parts) >= 5:
                try:
                    filings.append({
                        'cik': parts[0].strip(),
                        'company_name': parts[1].strip(),
                        'form_type': parts[2].strip(),
                        'date_filed': parts[3].strip(),
                        'file_name': parts[4].strip()
                    })
                except Exception:
                    continue
        
        return filings
    
    def phase_a_discovery(self, years: List[int] = None, quarters: List[int] = None) -> List[Dict]:
        logger.info("PHASE A: DISCOVERY")
        
        if years is None:
            current_year = datetime.now().year
            years = list(range(current_year - 2, current_year + 1))
        
        if quarters is None:
            quarters = [1, 2, 3, 4]
        
        cik_map = self.map_all_tickers_to_ciks()
        target_ciks = set(cik_map.values())
        
        if not target_ciks:
            logger.error("No CIKs found")
            return []
        
        logger.info(f"Target CIKs: {target_ciks}")
        all_10k_filings = []
        
        for year in years:
            for quarter in quarters:
                logger.info(f"Processing {year} Q{quarter}...")
                url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.idx"
                response = self._make_request(url)
                
                if not response:
                    logger.warning(f"Failed to fetch {year} Q{quarter}")
                    continue
                
                filings = self.parse_master_index(response.text)
                logger.info(f"Parsed {len(filings)} total filings from {year} Q{quarter}")
                
                for filing in filings:
                    # Normalize CIK for comparison (remove leading zeros)
                    filing_cik_normalized = filing['cik'].zfill(10)
                    
                    # Check if form type contains "10-K" (handles 10-K, 10-K/A, etc.)
                    if '10-K' in filing['form_type'] and filing_cik_normalized in target_ciks:
                        ticker = [t for t, c in cik_map.items() if c == filing_cik_normalized][0]
                        filing['ticker'] = ticker
                        filing['cik'] = filing_cik_normalized  # Update to padded version
                        all_10k_filings.append(filing)
                        logger.info(f"Found: {filing['company_name']} ({ticker}) - {filing['form_type']} - {filing['date_filed']}")
        
        logger.info(f"Found {len(all_10k_filings)} 10-K filings")
        return all_10k_filings
    
    
    def extract_accession_number(self, file_name: str) -> Optional[str]:
        match = re.search(r'edgar/data/\d+/(\d{10}-\d{2}-\d{6})', file_name)
        return match.group(1) if match else None
    
    def phase_b_mapping(self, filings: List[Dict]) -> List[Dict]:
        logger.info("PHASE B: MAPPING")
        enriched_filings = []
        
        for filing in filings:
            accession_number = self.extract_accession_number(filing['file_name'])
            if not accession_number:
                continue
            
            filing['accession_number'] = accession_number
            
            cik = filing['cik']
            acc_no_clean = accession_number.replace('-', '')
            filing['filing_url'] = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_clean}/{accession_number}-index.htm"
            
            acc_year = accession_number.split('-')[1]
            filing['fiscal_year'] = f"20{acc_year}" if acc_year else filing['date_filed'][:4]
            
            enriched_filings.append(filing)
        
        logger.info(f"Enriched {len(enriched_filings)} filings")
        return enriched_filings
    

    def phase_c_download(self, filings: List[Dict]) -> List[Dict]:
        logger.info("PHASE C: DOWNLOAD")
        downloaded_filings = []
        
        for idx, filing in enumerate(filings, 1):
            ticker = filing.get('ticker', 'UNKNOWN')
            accession_number = filing.get('accession_number', '')
            cik = filing['cik'].lstrip('0')
            
            logger.info(f"Downloading {idx}/{len(filings)}: {ticker} - {accession_number}")
            
            acc_no_clean = accession_number.replace('-', '')
            
            # APPROACH 1: Try to get the complete submission text file first
            # This contains all documents in a parseable format with <DOCUMENT> tags
            txt_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_clean}/{accession_number}.txt"
            logger.info(f"  Trying complete submission text file: {txt_url}")
            
            response = self._make_request(txt_url)
            
            if response and len(response.text) > 10000:
                content = response.text
                
                # Parse the document tags to find the main 10-K document
                doc_start = re.compile(r"<DOCUMENT>")
                doc_end = re.compile(r"</DOCUMENT>")
                doc_type = re.compile(r"<TYPE>([^\n]+)")
                
                # Find all document sections
                doc_starts = [m.end() for m in doc_start.finditer(content)]
                doc_ends = [m.start() for m in doc_end.finditer(content)]
                doc_types = [m.group(1).strip() for m in doc_type.finditer(content)]
                
                # Look for the main 10-K document (not exhibits)
                main_doc_content = None
                main_doc_type = None
                
                for start, end, dtype in zip(doc_starts, doc_ends, doc_types):
                    # We want 10-K, not EX-* (exhibits)
                    if '10-K' in dtype and not dtype.startswith('EX-'):
                        main_doc_content = content[start:end]
                        main_doc_type = dtype
                        logger.info(f"  Found main document: {dtype}")
                        break
                
                if main_doc_content:
                    # Save the extracted document
                    ticker_dir = self.output_dir / ticker
                    ticker_dir.mkdir(parents=True, exist_ok=True)
                    
                    filename = f"{ticker}_{accession_number.replace('-', '_')}_10k.txt"
                    filepath = ticker_dir / filename
                    
                    with open(filepath, 'w', encoding='utf-8', errors='ignore') as f:
                        f.write(main_doc_content)
                    
                    filing['download_path'] = str(filepath)
                    filing['file_size'] = len(main_doc_content)
                    filing['download_url'] = txt_url
                    filing['downloaded_at'] = datetime.now().isoformat()
                    filing['document_type'] = main_doc_type
                    
                    downloaded_filings.append(filing)
                    logger.info(f"  Saved: {filepath} ({len(main_doc_content):,} bytes)")
                    continue
            
            # APPROACH 2: If text file approach fails, try the HTML/iXBRL approach
            logger.info(f"  Text file approach failed, trying HTML index...")
            
            index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_clean}/{accession_number}-index.htm"
            response = self._make_request(index_url)
            
            if not response:
                logger.warning(f"Failed to fetch index for {accession_number}")
                continue
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find the primary document from the index
            doc_url = None
            primary_doc_name = None
            
            # Look for "Document Format Files" table
            tables = soup.find_all('table', {'summary': 'Document Format Files'})
            if not tables:
                tables = soup.find_all('table')
            
            for table in tables:
                rows = table.find_all('tr')
                for row in rows:
                    cells = row.find_all('td')
                    if len(cells) >= 4:
                        # Table: [Seq, Description, Document, Type, Size]
                        link_cell = cells[2] if len(cells) > 2 else None
                        type_cell = cells[3].get_text().strip().upper() if len(cells) > 3 else ''
                        
                        if link_cell:
                            link = link_cell.find('a', href=True)
                            if link:
                                href = link['href']
                                
                                # Find primary 10-K document
                                is_html = href.lower().endswith(('.htm', '.html'))
                                is_primary = type_cell == '10-K' or type_cell == '10-K/A'
                                is_not_exhibit = not type_cell.startswith('EX-')
                                
                                if is_html and is_primary and is_not_exhibit:
                                    primary_doc_name = href.split('/')[-1]
                                    
                                    # Try direct URL first
                                    if href.startswith('http'):
                                        doc_url = href
                                    elif href.startswith('/'):
                                        doc_url = f"https://www.sec.gov{href}"
                                    else:
                                        doc_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{acc_no_clean}/{primary_doc_name}"
                                    
                                    logger.info(f"  Found primary document: {primary_doc_name}")
                                    break
                    
                    if doc_url:
                        break
            
            if not doc_url:
                logger.warning(f"Could not find primary 10-K document for {accession_number}")
                continue
            
            # Download the HTML document
            logger.info(f"  Downloading from: {doc_url}")
            response = self._make_request(doc_url)
            
            if not response:
                logger.warning(f"Failed to download 10-K for {accession_number}")
                continue
            
            content = response.text
            
            # Validate content
            if len(content) < 5000:
                logger.warning(f"  Content too small ({len(content)} bytes)")
                continue
            
            # Save the file
            ticker_dir = self.output_dir / ticker
            ticker_dir.mkdir(parents=True, exist_ok=True)
            
            # Determine extension based on content
            if content.strip().startswith('<?xml') or '<xbrl' in content.lower()[:1000]:
                ext = 'xml'
            else:
                ext = 'htm'
            
            filename = f"{ticker}_{accession_number.replace('-', '_')}_10k.{ext}"
            filepath = ticker_dir / filename
            
            with open(filepath, 'w', encoding='utf-8', errors='ignore') as f:
                f.write(content)
            
            filing['download_path'] = str(filepath)
            filing['file_size'] = len(content)
            filing['download_url'] = doc_url
            filing['downloaded_at'] = datetime.now().isoformat()
            filing['primary_document'] = primary_doc_name
            
            downloaded_filings.append(filing)
            logger.info(f"  Saved: {filepath} ({len(content):,} bytes)")
        
        logger.info(f"Downloaded {len(downloaded_filings)} files")
        return downloaded_filings

    def phase_d_extraction(self, filings: List[Dict]) -> List[Dict]:
        logger.info("PHASE D: TEXT EXTRACTION")
        extracted_filings = []
        
        for idx, filing in enumerate(filings, 1):
            download_path = filing.get('download_path')
            if not download_path or not Path(download_path).exists():
                logger.warning(f"File not found for extraction: {download_path}")
                continue
                
            ticker = filing.get('ticker', 'UNKNOWN')
            logger.info(f"Extracting text {idx}/{len(filings)}: {ticker}")
            
            try:
                text_content = self.extract_text_from_10k(download_path)
                
                # Create new filename
                original_path = Path(download_path)
                clean_filename = original_path.stem + "_clean.txt"
                clean_path = original_path.parent / clean_filename
                
                with open(clean_path, 'w', encoding='utf-8') as f:
                    f.write(text_content)
                
                filing['clean_text_path'] = str(clean_path)
                filing['clean_text_size'] = len(text_content)
                extracted_filings.append(filing)
                logger.info(f"  Saved clean text: {clean_path}")
                
            except Exception as e:
                logger.error(f"Failed to extract text from {download_path}: {e}")
                
        logger.info(f"Extracted text for {len(extracted_filings)} filings")
        return extracted_filings


    def save_metadata(self, filings: List[Dict], format: str = 'json'):
        metadata_file = self.output_dir / f"metadata.{format}"
        
        if format == 'json':
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(filings, f, indent=2, ensure_ascii=False)
        elif format == 'csv':
            if not filings:
                return
            
            fieldnames = ['ticker', 'cik', 'company_name', 'form_type', 'date_filed', 
                         'accession_number', 'fiscal_year', 'filing_url', 'download_url',
                         'download_path', 'clean_text_path', 'file_size', 'clean_text_size', 'downloaded_at']
            
            with open(metadata_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                writer.writeheader()
                writer.writerows(filings)
        
        logger.info(f"Metadata saved to {metadata_file}")
    
    def run(self, years: List[int] = None, quarters: List[int] = None, save_format: str = 'json'):
        filings = self.phase_a_discovery(years=years, quarters=quarters)
        if not filings:
            return
        
        enriched_filings = self.phase_b_mapping(filings)
        if not enriched_filings:
            return
        
        downloaded_filings = self.phase_c_download(enriched_filings)
        if not downloaded_filings:
            return

        extracted_filings = self.phase_d_extraction(downloaded_filings)
        if not extracted_filings:
            return
        
        self.save_metadata(extracted_filings, format=save_format)
        logger.info(f"Complete! Processed {len(extracted_filings)} filings")

    def extract_text_from_10k(self, filepath: str) -> str:
        """
        Helper method to extract clean text from downloaded 10-K files.
        Handles both .txt (with <DOCUMENT> tags) and .htm formats.
        
        Args:
            filepath: Path to the downloaded 10-K file
            
        Returns:
            Extracted text content
        """
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        # Check if it contains <TEXT> tags (standard SEC SGML wrapper)
        # Often .txt files from EDGAR are SGML wrappers around HTML or plain text
        text_match = re.search(r'<TEXT>(.*?)</TEXT>', content, re.DOTALL | re.IGNORECASE)
        if text_match:
            content = text_match.group(1)
        
        # Use BeautifulSoup to clean up HTML/XML tags
        soup = BeautifulSoup(content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(['script', 'style']):
            script.decompose()
        
        # Get text
        text = soup.get_text()
        
        # Clean up whitespace
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
def main():
    scraper = SECFilingScraper()
    scraper.run(save_format='json')


if __name__ == "__main__":
    main()