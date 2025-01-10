import csv
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import subprocess
import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraping.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, min_delay: int = 1, max_delay: int = 5):
        self.min_delay = min_delay
        self.max_delay = max_delay
        self.last_request_time = 0

    def wait(self) -> None:
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < self.min_delay:
            delay = random.uniform(self.min_delay, self.max_delay)
            time.sleep(delay)
        self.last_request_time = time.time()

class URLScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (compatible; CRA-data-usability-script/v1.0/0.00; +; )'
        }
        self.rate_limiter = RateLimiter()
        self.allowed_tags = ['h1', 'h2', 'h3', 'h4', 'p']

    def validate_url(self, url: str) -> bool:
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except Exception:
            return False

    def scrape_url(self, url: str) -> str:
        if not self.validate_url(url):
            return 'Invalid URL format'

        try:
            self.rate_limiter.wait()
            response = requests.get(url, headers=self.headers, allow_redirects=False, timeout=30)
            
            if response.status_code == 301:
                return 'redirected'
            elif response.status_code != 200:
                return f'Failed to fetch content - Status code: {response.status_code}'

            soup = BeautifulSoup(response.text, 'html.parser')
            main_element = self._find_main_element(soup)
            
            if not main_element:
                return 'Main element not found on the page'

            return self._extract_content(main_element)

        except requests.Timeout:
            return 'Request timed out'
        except requests.RequestException as e:
            return f'Request failed: {str(e)}'
        except Exception as e:
            logger.error(f'Error scraping {url}: {str(e)}')
            return str(e)

    def _find_main_element(self, soup: BeautifulSoup) -> Optional[BeautifulSoup]:
        main_selectors = [
            {'attrs': {'property': 'mainContentOfPage', 'resource': '#wb-main', 'typeof': 'WebPageElement'}},
            {'attrs': {'property': 'mainContentOfPage', 'resource': '#wb-main', 'typeof': 'WebPageElement', 'class': 'col-md-9 col-md-push-3'}},
            {'attrs': {'role': 'main', 'property': 'mainContentOfPage', 'class': 'container'}}
        ]

        for selector in main_selectors:
            elements = soup.find_all('main', **selector)
            for element in elements:
                if 'class' not in element.attrs or 'container' in element.attrs.get('class', []):
                    return element
        return None

    def _extract_content(self, main_element: BeautifulSoup) -> str:
        unwanted_sections = [
            'provisional most-requested-bullets well well-sm brdr-0',
            'pagedetails container',
            'lnkbx',
            'pagedetails',
            'gc-prtts',
            'alert alert-info'
        ]
        
        for unwanted_class in unwanted_sections:
            sections = main_element.find_all('section', class_=unwanted_class)
            for section in sections:
                section.decompose()

        h2_elements = main_element.find_all('h2', class_='h3', string=lambda text: "On this page:" in text)
        for h2 in h2_elements:
            next_sibling = h2.find_next_sibling()
            if next_sibling and next_sibling.name == 'ul':
                h2.decompose()
                next_sibling.decompose()

        scraped_content = []
        for tag in self.allowed_tags:
            elements = main_element.find_all(tag)
            for element in elements:
                if tag == 'h2' and any(text in element.get_text() for text in ['Chat with Charlie', 'Clavardez avec Charlie']):
                    continue
                scraped_content.append(element.get_text().strip())

        return ' '.join(scraped_content)[:2500]

class CSVProcessor:
    def __init__(self, input_file: str, output_file: str):
        self.input_file = Path(input_file)
        self.output_file = Path(output_file)
        self.scraper = URLScraper()

    def validate_files(self) -> None:
        if not self.input_file.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_file}")
        if not self.input_file.suffix == '.csv':
            raise ValueError("Input file must be a CSV file")
        
        with open(self.input_file, 'r', encoding='utf-8') as infile:
            reader = csv.reader(infile)
            header = next(reader, None)
            if not header or header[0].lower() != 'urls':
                raise ValueError("CSV file must have 'urls' as the header in the first column")

    def process(self) -> None:
        try:
            self.validate_files()
            output_data = []
            total_urls = sum(1 for _ in open(self.input_file)) - 1

            urls = []
            with open(self.input_file, 'r', newline='', encoding='utf-8') as infile:
                reader = csv.reader(infile)
                next(reader)
                logger.info("Processing URLs from CSV (header: 'urls')")
                for row in reader:
                    if row and row[0].strip():
                        urls.append(row[0].strip())

            for i, url in enumerate(urls):
                    logger.info(f'Processing URL {i + 1}/{total_urls}: {url}')
                    
                    scraped_content = self.scraper.scrape_url(url)
                    logger.info(f'Adding to output data - URL: {url}')
                    output_data.append({'url': url, 'scraped_content': scraped_content})
                    logger.info(f'Current output data size: {len(output_data)}')

            logger.info(f'Writing {len(output_data)} URLs to CSV')
            with open(self.output_file, 'w', newline='', encoding='utf-8') as outfile:
                writer = csv.DictWriter(outfile, fieldnames=['url', 'scraped_content'])
                writer.writeheader()
                for data in output_data:
                    writer.writerow(data)
            logger.info(f'Scraped data saved to {self.output_file}')

            self._run_metadata_generator()

        except Exception as e:
            logger.error(f"Error processing CSV: {str(e)}")
            raise

    def _run_metadata_generator(self) -> None:
        try:
            logger.info("Starting metadata generation...")
            result = subprocess.run(
                [sys.executable, 'metadata_generator.py'],
                capture_output=True,
                text=True,
                check=True
            )
            logger.info("Metadata generation completed successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Metadata generation failed: {e.stderr}")
            raise
        except Exception as e:
            logger.error(f"Error running metadata generator: {str(e)}")
            raise

def main():
    try:
        processor = CSVProcessor('inputtoscrape.csv', 'scraped_content.csv')
        processor.process()
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
