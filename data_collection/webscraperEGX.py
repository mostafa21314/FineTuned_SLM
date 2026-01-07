import os
import re
import time
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from urllib.parse import urljoin

try:
    import cloudscraper
    HAS_CLOUDSCRAPER = True
except ImportError:
    HAS_CLOUDSCRAPER = False

try:
    from playwright.sync_api import sync_playwright
    HAS_PLAYWRIGHT = True
except ImportError:
    HAS_PLAYWRIGHT = False

class EGXFinancialScraper:
    def __init__(self, output_dir="data/raw/pdfs"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        if HAS_CLOUDSCRAPER:
            self.session = cloudscraper.create_scraper()
            print("[INFO] Using cloudscraper for bot bypass.")
        else:
            self.session = requests.Session()
            self.session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            })
            print("[WARNING] cloudscraper not found. CIB scanning may fail due to bot protection.")

        
        self.targets = [
             {
                "name": "CIB",
                "url": "https://www.cibeg.com/en/investor-relations/ir-library/financial-statements",
                "parser": self.parse_cib
            },

            {
                "name": "CAE",
                "url": "https://www.ca-egypt.com/en/investor-relation/financial-information/financial-statements/financial-statement/?bank_segment=personal-banking",
                "parser": self.parse_cae
            },
            {
                "name": "ADIB",
                "url": "https://www.adib.eg/investor-relations/financial-results",
                "parser": self.parse_adib
            }
        ]
        
        self.years = list(range(2023, 2026)) # 2023 to 2025

    def download_file(self, url, bank_name, year, filename):
        try:
            # Create directory: data/raw/pdfs/{Bank}/{Year}
            year_dir = self.output_dir / bank_name / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)
            
            # Clean filename
            safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
            if not safe_filename.lower().endswith('.pdf'):
                safe_filename += '.pdf'
                
            filepath = year_dir / safe_filename
            
            # Allow skipping if exists
            if filepath.exists():
                print(f"[SKIP] {filepath} already exists")
                return True
            
            print(f"[DOWNLOADING] {url} -> {filepath}")
            response = self.session.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"[SUCCESS] Saved to {filepath}")
            time.sleep(1) # Polite delay
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to download {url}: {e}")
            return False

    def parse_cae(self, html, bank_data):
        """
        Parser for Credit Agricole Egypt
        Target: <a href="..."><strong>CAE financials QX YYYY Consolidated</strong></a>
        """
        soup = BeautifulSoup(html, 'html.parser')
        links = soup.find_all('a', href=True)
        count = 0
        
        for link in links:
            text = link.get_text(" ", strip=True).lower()
            href = link['href']
            
            # Filter for consolidated
            if "consolidated" not in text and "consolidated" not in href.lower():
                continue

            # Find year
            year_match = re.search(r'202[0-5]', text)
            if not year_match:
                year_match = re.search(r'202[0-5]', href)
            
            if year_match:
                year = int(year_match.group(0))
                if year in self.years:
                    full_url = urljoin(bank_data['url'], href)
                    
                    # Try to extract Quarter info if possible
                    q_match = re.search(r'q[1-4]', text)
                    q_str = q_match.group(0) if q_match else "Annual"
                    
                    name = f"CAE_{year}_{q_str}_Consolidated.pdf"
                    if self.download_file(full_url, "CAE", year, name):
                        count += 1
        
        print(f"[CAE] Found {count} consolidated reports")

    def parse_adib(self, html, bank_data):
        """
        Parser for ADIB Egypt
        Target: <div class="finance annualfinance" name="Consolidated"> ... </div>
        Checks for 'annualfinance' class or quarter keywords to distinguish reports.
        """
        soup = BeautifulSoup(html, 'html.parser')
        divs = soup.select('div.finance')
        count = 0
        
        for div in divs:
            div_text = div.get_text(" ", strip=True).lower()
            
            # Check name attribute or text for "Consolidated"
            name_attr = div.get('name', '').lower()
            if "consolidated" not in name_attr and "consolidated" not in div_text:
                continue
            
            # Year logic
            year_match = re.search(r'202[0-5]', div_text)
            if not year_match:
                continue
            year = int(year_match.group(0))
            if year not in self.years:
                continue

            # Determine Period
            classes = div.get('class', [])
            period = "Annual" # Default preference if annualfinance is present
            
            if 'annualfinance' in classes:
                period = "Annual"
            else:
                # Detect quarters
                if "q1" in div_text or "march" in div_text or "first quarter" in div_text:
                    period = "Q1"
                elif "q2" in div_text or "june" in div_text or "second quarter" in div_text:
                    period = "Q2"
                elif "q3" in div_text or "september" in div_text or "third quarter" in div_text:
                    period = "Q3"
                elif "q4" in div_text or "december" in div_text or "fourth quarter" in div_text:
                    period = "Q4"
            
            # Link logic
            a_tag = div.find('a', href=True)
            if a_tag:
                full_url = urljoin(bank_data['url'], a_tag['href'])
                name = f"ADIB_{year}_{period}_Consolidated.pdf"
                if self.download_file(full_url, "ADIB", year, name):
                    count += 1
                
        print(f"[ADIB] Found {count} consolidated reports")

    def parse_cib(self, html, bank_data):
        """
        Parser for CIB Egypt using Playwright to bypass bot protection
        """
        if not HAS_PLAYWRIGHT:
            print("[ERROR] Playwright not installed. Install with: pip install playwright && playwright install")
            return
        
        print("[CIB] Using Playwright to bypass bot protection and render JavaScript...")
        count = 0
        
        with sync_playwright() as p:
            # Launch browser (headless mode)
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = context.new_page()
            
            try:
                print(f"[CIB] Loading page: {bank_data['url']}")
                page.goto(bank_data['url'], wait_until='networkidle', timeout=60000)
                
                # Wait for content to load (adjust selector based on actual page)
                print("[CIB] Waiting for content to load...")
                time.sleep(5)  # Give JavaScript time to render
                
                # Get all links
                links = page.locator('a[href]').all()
                print(f"[CIB] Found {len(links)} total links")
                
                for link in links:
                    try:
                        href = link.get_attribute('href')
                        if not href:
                            continue
                            
                        # Make absolute URL
                        if not href.startswith('http'):
                            href = urljoin(bank_data['url'], href)
                        
                        href_lower = href.lower()
                        
                        # Check if PDF
                        if not href_lower.endswith('.pdf'):
                            continue
                        
                        # Check for consolidated
                        text = link.inner_text().lower() if link.inner_text() else ""
                        is_consolidated = ("consolidated" in href_lower or 
                                         "consolidated" in text or
                                         ("financial" in href_lower and "statements" in href_lower and "separate" not in href_lower))
                        
                        if not is_consolidated:
                            continue
                        
                        # Extract year
                        year_match = re.search(r'/202[0-5]/', href_lower)
                        if not year_match:
                            year_match = re.search(r'202[0-5]', href_lower)
                        
                        if not year_match:
                            continue
                        
                        year_str = year_match.group(0).replace('/', '')
                        year = int(year_str)
                        
                        if year not in self.years:
                            continue
                        
                        # Extract quarter
                        q_str = "Annual"
                        if "/q4/" in href_lower or "december" in href_lower:
                            q_str = "Q4"
                        elif "/q3/" in href_lower or "september" in href_lower:
                            q_str = "Q3"
                        elif "/q2/" in href_lower or "june" in href_lower:
                            q_str = "Q2"
                        elif "/q1/" in href_lower or "march" in href_lower:
                            q_str = "Q1"
                        
                        # Download using Playwright to bypass bot protection
                        filename = f"CIB_{year}_{q_str}_Consolidated.pdf"
                        if self.download_cib_with_playwright(page, href, year, filename):
                            count += 1
                            
                    except Exception as e:
                        continue
                
            except Exception as e:
                print(f"[ERROR] Playwright navigation failed: {e}")
            finally:
                browser.close()
        
        print(f"[CIB] Found {count} consolidated reports")

    def download_cib_with_playwright(self, page, url, year, filename):
        """Download CIB PDF using Playwright context to maintain session/bypass bot protection"""
        try:
            year_dir = self.output_dir / "CIB" / str(year)
            year_dir.mkdir(parents=True, exist_ok=True)
            
            safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
            if not safe_filename.lower().endswith('.pdf'):
                safe_filename += '.pdf'
            
            filepath = year_dir / safe_filename
            
            if filepath.exists():
                print(f"[SKIP] {filepath} already exists")
                return True
            
            print(f"[DOWNLOADING] {url} -> {filepath}")
            
            # Navigate to PDF in same context (maintains cookies/session)
            response = page.goto(url, wait_until='commit', timeout=60000)
            
            if response and response.ok:
                # Get the PDF content
                pdf_content = response.body()
                
                # Verify it's a PDF
                if pdf_content[:4] != b'%PDF':
                    print(f"[ERROR] Downloaded content is not a PDF")
                    return False
                
                # Save the file
                with open(filepath, 'wb') as f:
                    f.write(pdf_content)
                
                print(f"[SUCCESS] Saved to {filepath}")
                time.sleep(2)  # Polite delay
                return True
            else:
                print(f"[ERROR] Failed to download - Status: {response.status if response else 'No response'}")
                return False
                
        except Exception as e:
            print(f"[ERROR] Playwright download failed for {url}: {e}")
            return False

    def run(self):
        print("Starting EGX Scraper...")
        print(f"Playwright available: {HAS_PLAYWRIGHT}")
        print(f"Cloudscraper available: {HAS_CLOUDSCRAPER}")
        
        for bank in self.targets:
            print(f"\nProcessing {bank['name']}...")
            try:
                if bank['name'] == "CIB":
                    # Use Playwright for CIB
                    bank['parser'](html=None, bank_data=bank)
                else:
                    # Use requests for CAE and ADIB (unchanged)
                    response = self.session.get(bank['url'], timeout=30)
                    response.raise_for_status()
                    bank['parser'](html=response.text, bank_data=bank)
            except Exception as e:
                print(f"[ERROR] Failed to process {bank['name']}: {e}")

if __name__ == "__main__":
    scraper = EGXFinancialScraper()
    scraper.run()