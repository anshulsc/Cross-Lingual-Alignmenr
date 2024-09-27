# scripts/download_wiki.py

import os
import wget
import logging
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

def setup_logging(log_file='logs/download.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        filename=log_file,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

def download_wikipedia_dump(language_code, url, dest_folder):
    os.makedirs(dest_folder, exist_ok=True)
    dest_path = os.path.join(dest_folder, f"{language_code}wiki-latest-pages-articles.xml.bz2")
    if not os.path.exists(dest_path):
        logging.info(f"Starting download for {language_code} Wikipedia.")
        try:
            wget.download(url, out=dest_path)
            logging.info(f"Downloaded {language_code} Wikipedia dump to {dest_path}.")
        except Exception as e:
            logging.error(f"Failed to download {language_code} Wikipedia dump: {e}")
            raise
    else:
        logging.info(f"{dest_path} already exists. Skipping download.")

def main(config):
    setup_logging()

    with ThreadPoolExecutor(max_workers=4) as executor:  
        future_to_lang = {executor.submit(download_wikipedia_dump, lang, details['url'], details['destination']): lang 
                          for lang, details in config['wikipedia_dumps'].items()}
        
        for future in as_completed(future_to_lang):
            lang = future_to_lang[future]
            try:
                future.result()  # This will raise any exceptions that occurred during download
                logging.info(f"Completed download for {lang} Wikipedia.")
            except Exception as e:
                logging.error(f"Error downloading {lang} Wikipedia: {e}")

if __name__ == "__main__":
    with open('configs/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    main(config)
