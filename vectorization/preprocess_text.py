import os
import re
import logging
import nltk
from indicnlp.tokenize import indic_tokenize
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(log_file='logs/extract.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def clean_text(text, language):
    """
    Cleans the input text by removing unwanted characters and normalizing it.

    Parameters:
    - text (str): The raw text to clean.
    - language (str): 'en' for English, 'hi' for Hindi.

    Returns:
    - str: The cleaned text.
    """
    # Remove multiple newlines and replace with space
    text = re.sub(r'\n+', ' ', text)
    
    # Remove content within parentheses and brackets
    text = re.sub(r'\(.*?\)', '', text)
    text = re.sub(r'\[.*?\]', '', text)
    
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    
    # Remove punctuation and numbers (except for Hindi characters)
    if language == 'en':
        # For English: Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
    elif language == 'hi':
        # For Hindi: Remove punctuation and numbers, retain Devanagari script
        text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    else:
        raise ValueError("Unsupported language")
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def tokenize_text(text, language):
    """
    Tokenizes the input text based on the specified language.

    Parameters:
    - text (str): The cleaned text to tokenize.
    - language (str): 'en' for English, 'hi' for Hindi.

    Returns:
    - list: List of tokens.
    """
    if language == 'en':
        tokens = nltk.word_tokenize(text)
    elif language == 'hi':
        tokens = indic_tokenize.trivial_tokenize(text)
    else:
        raise ValueError("Unsupported language")
    return tokens

def preprocess_articles(input_file, output_file, language):
    """
    Preprocesses articles by cleaning and tokenizing.

    Parameters:
    - input_file (str): Path to the input file containing raw articles.
    - output_file (str): Path to save the preprocessed articles.
    - language (str): 'en' for English, 'hi' for Hindi.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            if '\t' in line:
                parts = line.strip().split('\t', 1)
                # Ensure we have exactly two parts (title and text)
                if len(parts) == 2:
                    title, text = parts
                else:
                    continue  # Skip if the line doesn't have the correct format
            else:
                # If no tab character, treat the entire line as text
                text = line.strip()

            # Ensure text is a string before processing
            if isinstance(text, list):
                text = ' '.join(text)

            cleaned_text = clean_text(text, language)
            tokens = tokenize_text(cleaned_text, language)

            # Join tokens with space
            processed_text = ' '.join(tokens)
            outfile.write(processed_text + '\n')

def process_language(lang, config):
    """
    Process a single language by calling preprocess_articles function.

    Parameters:
    - lang (str): Language code (e.g., 'en' or 'hi').
    - config (dict): Configuration dictionary.
    """
    preprocess_config = config['data_preparation']
    processed_data_dir = preprocess_config['processed_data_dir']
    
    input_file = os.path.join(processed_data_dir, f"final_{lang}.txt")
    output_file = os.path.join(processed_data_dir, f"preprocessed_{lang}.txt")
    
    logging.info(f"Preprocessing {lang} articles...")
    preprocess_articles(input_file, output_file, language=lang)
    logging.info(f"Preprocessed {lang} articles saved to {output_file}")

def main():
    config = load_config("configs/config.yaml")
    setup_logging()

    languages = ['en', 'hi']

    # Sequentially process each language
    for lang in languages:
        process_language(lang, config)

if __name__ == "__main__":
    main()
