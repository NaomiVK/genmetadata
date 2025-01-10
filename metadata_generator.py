import csv
import logging
from openai import OpenAI
from groq import Groq
import langdetect
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterator
import backoff
from pathlib import Path
import sys
import json
from tqdm import tqdm
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('metadata_script.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    model_name: str
    review_model_name: str
    
    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> 'Config':
        default_config = {
            "model_name": "mixtral-8x7b-32768",
            "review_model_name": "llama-3.3-70b-versatile"
        }
        
        if config_path:
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                    default_config.update(config_data)
            except Exception as e:
                logger.warning(f"Failed to load config file: {e}. Using defaults.")
        
        return cls(**default_config)

class ContentProcessor:
    def __init__(self, text: str):
        self.text = text.strip()
    
    def is_valid(self) -> bool:
        return bool(self.text) and len(self.text) >= 50

    @staticmethod
    def detect_language(text: str) -> str:
        try:
            return langdetect.detect(text)
        except langdetect.lang_detect_exception.LangDetectException:
            return 'en'

class MetadataProcessor:
    def __init__(self, config: Config):
        self.config = config
        self.client = Groq(
            api_key=os.getenv("GROQ_API_KEY")
        )

    @staticmethod
    def truncate_summary(summary: str, max_length: int = 250) -> str:
        if len(summary) > max_length:
            last_full_stop = summary.rfind('.', 0, max_length)
            return summary[:last_full_stop + 1] if last_full_stop != -1 else summary[:max_length]
        return summary

    @backoff.on_exception(
        backoff.expo,
        (Exception,),
        max_tries=3,
        on_backoff=lambda details: logger.warning(f"Retrying API call: {details}")
    )
    def _make_completion_call(
        self,
        prompt_text: str,
        max_tokens: int,
        temperature: float,
        model_name: str = None
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model_name or self.config.model_name,
                messages=[{"role": "user", "content": prompt_text}],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content.strip() if response.choices else ""
        except Exception as e:
            logger.error(f"API call failed: {str(e)}")
            raise

    def summarize_content(self, content: str) -> str:
        processor = ContentProcessor(content)
        if not processor.is_valid():
            return "Content too short or invalid for summarization"

        language = processor.detect_language(content)
        
        prompt_text = (
            "As a search engine optimization expert, using natural language processing, provide a concise, complete summary suitable "
            "for a meta description in English. The summary should be informative, use topic-specific terms, in full sentences, "
            "and must end concisely within 275 characters. Avoid using ellipses or cutting off sentences.\n\n"
            f"{content}\n\nSummary:"
        ) if language == 'en' else (
            "À l'aide du traitement automatique du langage naturel, fournissez un résumé concis et complet adapté "
            "à une méta-description en français. Le résumé doit être informatif, en phrases complètes, "
            "et doit se terminer de manière concise dans les 275 caractères. Évitez l'utilisation de points de suspension ou de coupures abruptes.\n\n"
            f"{content}\n\nRésumé:"
        )
        
        summary = self._make_completion_call(
            prompt_text=prompt_text,
            max_tokens=200,
            temperature=0.5,
            model_name=self.config.model_name
        )
        return self.truncate_summary(summary)

    def generate_keywords(self, content: str) -> str:
        processor = ContentProcessor(content)
        if not processor.is_valid():
            return "Content too short or invalid for keyword generation"

        prompt_text = (
            "As a search engine optimization expert, identify and extract 10 meaningful, topic-specific meta keywords from the following content. "
            "Please list the keywords in a comma-separated format only. Do not include any additional notes, explanations, or commentary. "
            "Exclude 'Canada Revenue Agency' from the keywords. "
            "Focus strictly on providing the keywords.:\n\n"
            f"{content}\n\nKeywords:"
        )
        
        return self._make_completion_call(
            prompt_text=prompt_text,
            max_tokens=80,
            temperature=0.3,
            model_name=self.config.model_name
        )

    def review_metadata(self, content: str, description: str, keywords: str) -> str:
        logger.info(f"Starting metadata review with model: {self.config.review_model_name}")
        logger.info(f"Input Description: {description}")
        logger.info(f"Input Keywords: {keywords}")
        
        prompt_text = (
            "You are an expert Search Engine Optimization (SEO) specialist with extensive knowledge in metadata optimization. "
            "Review the following content and its metadata with a focus on maximizing search engine visibility and user engagement. "
            "Analyze the meta description and keywords for:\n"
            "1. Relevance to content\n"
            "2. Search engine optimization effectiveness\n"
            "3. Keyword density and placement\n"
            "4. User engagement potential\n"
            "5. Competitive advantage in search results\n\n"
            "Provide specific, actionable recommendations for improvements. If the metadata is already optimal, explain why.\n\n"
            f"Content: {content}\n\n"
            f"Current Meta Description: {description}\n\n"
            f"Current Keywords: {keywords}\n\n"
            "Expert SEO Analysis and Recommendations:"
        )
        
        try:
            recommendations = self._make_completion_call(
                prompt_text=prompt_text,
                max_tokens=400,  
                temperature=0.3,
                model_name=self.config.review_model_name
            )
            
            logger.info("Received recommendations from model")
            logger.info(f"Recommendations: {recommendations}")
            
            if not recommendations or len(recommendations.strip()) < 10:
                logger.error("Received empty or very short recommendations")
                return "Error: Model provided insufficient recommendations"
                
            return recommendations
            
        except Exception as e:
            logger.error(f"Error during metadata review: {str(e)}")
            return f"Error during review: {str(e)}"

class CSVHandler:
    def __init__(self, input_path: str, output_path: str):
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def validate_files(self) -> None:
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_path}")
        if not self.input_path.suffix == '.csv':
            raise ValueError("Input file must be a CSV file")

    def read_csv(self) -> Iterator[Dict]:
        try:
            with open(self.input_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                yield from reader
        except UnicodeDecodeError:
            with open(self.input_path, 'r', newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                yield from reader

    def write_row(self, writer: csv.DictWriter, row: Dict) -> None:
        try:
            writer.writerow(row)
        except Exception as e:
            logger.error(f"Error writing row: {e}")
            raise

def main():
    try:
        if not os.getenv("GROQ_API_KEY"):
            logger.error("GROQ_API_KEY environment variable is not set")
            sys.exit(1)
            
        config = Config.load_config()
        processor = MetadataProcessor(config)
        csv_handler = CSVHandler('scraped_content.csv', 'processed_metadata.csv')
        
        csv_handler.validate_files()
        
        total_rows = sum(1 for _ in csv_handler.read_csv())
        
        with open(csv_handler.output_path, 'w', newline='', encoding='utf-8') as outfile:
            fieldnames = ['url', 'scraped_content', 'generated_description', 'generated_keywords', 'model_recommendations']
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            
            with tqdm(total=total_rows, desc="Processing content") as pbar:
                for row in csv_handler.read_csv():
                    try:
                        scraped_content = row['scraped_content']
                        
                        description = processor.summarize_content(scraped_content)
                        keywords = processor.generate_keywords(scraped_content)
                        
                        recommendations = processor.review_metadata(scraped_content, description, keywords)
                        
                        output_row = {
                            'url': row['url'],
                            'scraped_content': scraped_content,
                            'generated_description': description,
                            'generated_keywords': keywords,
                            'model_recommendations': recommendations
                        }
                        
                        csv_handler.write_row(writer, output_row)
                        pbar.update(1)
                        
                    except Exception as e:
                        logger.error(f"Error processing row: {e}")
                        continue

        logger.info(f'Metadata generation completed. Results saved to {csv_handler.output_path}')
        
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
