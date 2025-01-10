import streamlit as st
import pandas as pd
from scrape_urls import URLScraper, RateLimiter
from metadata_generator import Config, MetadataProcessor, ContentProcessor
import tempfile
import os
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('streamlit_app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def process_urls(uploaded_file, progress_bar, status_text):
    df = pd.read_csv(uploaded_file)
    
    scraper = URLScraper()
    total_urls = len(df)
    
    output_data = []
    
    first_column = df.columns[0]
    
    for i, row in df.iterrows():
        url = row[first_column].strip()
        status_text.text(f'Processing URL {i+1}/{total_urls}: {url}')
        
        scraped_content = scraper.scrape_url(url)
        output_data.append({'url': url, 'scraped_content': scraped_content})
        
        progress_bar.progress((i + 1) / total_urls)
    
    return pd.DataFrame(output_data)

def generate_metadata(scraped_df, progress_bar, status_text, model_status):
    config = Config.load_config()
    processor = MetadataProcessor(config)
    
    total_rows = len(scraped_df)
    results = []
    
    for i, row in scraped_df.iterrows():
        model_status.text(f'Model being used: {config.model_name}')
        status_text.text(f'Generating metadata for URL {i+1}/{total_rows}')
        
        scraped_content = row['scraped_content']
        description = processor.summarize_content(scraped_content)
        keywords = processor.generate_keywords(scraped_content)
        
        model_status.text(f'Model: {config.review_model_name} is now reviewing metadata')
        status_text.text(f'Reviewing metadata for URL {i+1}/{total_rows}')
        
        with st.expander(f"Metadata for URL {i+1}", expanded=False):
            st.text("Generated Description:")
            st.code(description)
            st.text("Generated Keywords:")
            st.code(keywords)
            
            recommendations = processor.review_metadata(scraped_content, description, keywords)
            
            st.text("SEO Expert Recommendations:")
            st.code(recommendations)
        
        results.append({
            'url': row['url'],
            'scraped_content': scraped_content,
            'generated_description': description,
            'generated_keywords': keywords,
            'model_recommendations': recommendations
        })
        
        progress_bar.progress((i + 1) / total_rows)
    
    return pd.DataFrame(results)

def main():
    st.title('URL Content Scraper and Metadata Generator')
    
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY environment variable is not set")
        return
    
    st.info("Upload a CSV file with URLs in the first column. Please include a header called 'urls'")
    uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
    
    if uploaded_file is not None:
        st.subheader('URL Scraping Progress')
        scraping_progress = st.progress(0)
        scraping_status = st.empty()
        
        scraped_df = process_urls(uploaded_file, scraping_progress, scraping_status)
        
        st.success('URLs scraped successfully!')
        st.subheader('Generating Metadata')
        
        metadata_progress = st.progress(0)
        metadata_status = st.empty()
        model_status = st.empty()  
        
        final_df = generate_metadata(scraped_df, metadata_progress, metadata_status, model_status)
        
        csv = final_df.to_csv(index=False)
        st.success('Metadata generation complete!')
        st.download_button(
            label="Download processed metadata CSV",
            data=csv,
            file_name="processed_metadata.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    main()
