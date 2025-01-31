import streamlit as st
import pandas as pd
from scrape_urls import URLScraper, RateLimiter
from metadata_generator import Config, MetadataProcessor, ContentProcessor
import tempfile
import os
import logging
import sys
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

def validate_input_csv(df, input_type):
    if input_type == "urls_only":
        if len(df.columns) < 1:
            st.error("CSV file must contain at least one column with URLs")
            return False
        return True
    else:  # urls_and_content
        required_columns = ['url', 'scraped_content']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            st.error(f"CSV file must contain the following columns: {', '.join(missing_columns)}")
            return False
        return True

def process_urls(uploaded_file, progress_bar, status_text, input_type):
    df = pd.read_csv(uploaded_file)
    
    if not validate_input_csv(df, input_type):
        return None
        
    if input_type == "urls_only":
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
    else:  # urls_and_content
        # Rename columns if needed to match expected format
        df = df.rename(columns={
            df.columns[0]: 'url' if 'url' not in df.columns else df.columns[0],
            df.columns[1]: 'scraped_content' if 'scraped_content' not in df.columns else df.columns[1]
        })
        progress_bar.progress(1.0)
        status_text.text('Content already provided, proceeding to metadata generation')
        return df[['url', 'scraped_content']]

def generate_metadata(scraped_df, progress_bar, status_text, model_status):
    try:
        config = Config.load_config()
        processor = MetadataProcessor(config)
        
        total_rows = len(scraped_df)
        results = []
        
        for i, row in scraped_df.iterrows():
            try:
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
                    
                    st.text("SEO Recommendations:")
                    st.code(recommendations)
                
                results.append({
                    'url': row['url'],
                    'scraped_content': scraped_content,
                    'generated_description': description,
                    'generated_keywords': keywords,
                    'model_recommendations': recommendations
                })
                
                progress_bar.progress((i + 1) / total_rows)
                
            except Exception as e:
                logger.error(f"Error processing row {i+1}: {str(e)}")
                st.error(f"Error processing URL {row['url']}: {str(e)}")
                continue
        
        return pd.DataFrame(results)
        
    except Exception as e:
        logger.error(f"Error in metadata generation: {str(e)}")
        st.error(f"An error occurred during metadata generation: {str(e)}")
        return None

def main():
    # Initialize session state
    if 'processing_complete' not in st.session_state:
        st.session_state.processing_complete = False
        st.session_state.final_csv = None
    
    st.title('URL Content Scraper and Metadata Generator')
    
    if not os.getenv("GROQ_API_KEY"):
        st.error("GROQ_API_KEY environment variable is not set")
        return
    
    # Only show input options if processing is not complete
    if not st.session_state.processing_complete:
        input_type = st.radio(
            "Select input type:",
            ["urls_only", "urls_and_content"],
            format_func=lambda x: "URLs only (for scraping)" if x == "urls_only" else "URLs and pre-scraped content"
        )
        
        if input_type == "urls_only":
            st.info("Upload a CSV file with URLs in the first column. Please include a header called 'urls'")
        else:
            st.info("Upload a CSV file with two columns: 'url' and 'scraped_content'")
        
        uploaded_file = st.file_uploader("Choose CSV file", type=['csv'])
        
        if uploaded_file is not None:
            st.subheader('Processing Progress')
            scraping_progress = st.progress(0)
            scraping_status = st.empty()
            
            scraped_df = process_urls(uploaded_file, scraping_progress, scraping_status, input_type)
            
            if scraped_df is None:
                return
            
            st.success('URLs scraped successfully!')
            st.subheader('Generating Metadata')
            
            metadata_progress = st.progress(0)
            metadata_status = st.empty()
            model_status = st.empty()  
            
            final_df = generate_metadata(scraped_df, metadata_progress, metadata_status, model_status)
            
            if final_df is not None:
                csv_data = final_df.to_csv(index=False)
                
                # Clear progress indicators
                metadata_progress.empty()
                metadata_status.empty()
                model_status.empty()
                
                # Show completion state
                st.success('Metadata generation complete!')
                st.download_button(
                    label="Download processed metadata CSV",
                    data=csv_data,
                    file_name="processed_metadata.csv",
                    mime="text/csv"
                )
                st.success("Processing completed! You may download your results.")
                
                # Update session state to prevent reprocessing
                st.session_state.processing_complete = True
                st.session_state.final_csv = csv_data
            else:
                st.error("Failed to generate metadata. Please check the logs for details.")
    else:
        # Show saved completion state for subsequent visits
        st.success('Metadata generation complete!')
        st.download_button(
            label="Download processed metadata CSV",
            data=st.session_state.final_csv,
            file_name="processed_metadata.csv",
            mime="text/csv"
        )

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
