import os
import json
import bz2
import yaml
import logging
from concurrent.futures import ProcessPoolExecutor

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(log_file='logs/extract.log'):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def extract_first_n_articles(input_path, output_file, n=10000):
    
    # Read and write first n articles
    count = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for root, dirs, files in os.walk(input_path):
            for file in files:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            logging.info(f"Extracted {count} articles to {output_file}")
                            article = json.loads(line)
                            title = article.get('title', '')
                            text = article.get('text', '')
                            outfile.write(f"{title}\t{text}\n")
                            count += 1
                            if count >= n:
                                logging.info(f"Extracted {n} articles to {output_file}")
                                return
    logging.info(f"Extracted {count} articles to {output_file}")

def process_language(lang, dump_config, num_articles):
    input_dump = os.path.join(dump_config['extracted_data_dir'], f"extracted_{lang}")
    output_file = os.path.join(dump_config['processed_data_dir'], f"final_{lang}.txt")
    extract_first_n_articles(input_dump, output_file, n=num_articles)

def main():
    setup_logging(log_file='logs/extract.log')
    config = load_config("/Users/anshulsingh/lockedin/cross-lingual-alignment/configs/config.yaml")

    
    dump_config = config['data_preparation']
    num_articles = dump_config['num_articles']
    processed_data_dir = dump_config['processed_data_dir']
    os.makedirs(processed_data_dir, exist_ok=True)
    
    languages = ['en', 'hi']
    # Use ProcessPoolExecutor to parallelize the process
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_language, lang, dump_config, num_articles) for lang in languages]
        for future in futures:
            future.result()  # Ensure all processes are completed

if __name__ == "__main__":
    main()
