import os
import logging
import fasttext
import pickle
import numpy as np
import yaml
import heapq

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(log_file='logs/vectorization.log'):
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

def train_fasttext_model(input_file, output_model, config):
    logging.info(f"Training FastText model on {input_file}...")
    model = fasttext.train_unsupervised(
        input=input_file,
        model='skipgram',
        dim=config['dim'],
        epoch=config['epoch'],
        lr=config['lr'],
        wordNgrams=config['wordNgrams'],
        loss=config['loss'],
        bucket=config['bucket'],
        minn=config['minn'],
        maxn=config['maxn'],
        thread=config['thread']
    )
    model.save_model(output_model)
    logging.info(f"FastText model saved to {output_model}")
    return model

def load_model(model_path):
    model = fasttext.load_model(model_path)
    logging.info(f"Loaded FastText model from {model_path}")
    return model

def get_top_n_words(model, n=100000):
    words = model.get_words()
    frequencies = [model.get_word_frequency(word) for word in words]
    word_freq = list(zip(words, frequencies))
    top_n = heapq.nlargest(n, word_freq, key=lambda x: x[1])
    top_words = [word for word, freq in top_n]
    return top_words

def filter_embeddings(model, top_words, language, embeddings_dir):
    valid_words = []
    embeddings = []
    missing = 0
    for word in top_words:
        try:
            vec = model.get_word_vector(word)
            valid_words.append(word)
            embeddings.append(vec)
        except KeyError:
            missing += 1
            continue
    embeddings = np.array(embeddings)
    logging.info(f"{language.upper()}: {missing} words missing from the model.")
    
    # Save filtered embeddings
    filtered_path = os.path.join(embeddings_dir, f"filtered_embeddings_{language}.pkl")
    with open(filtered_path, 'wb') as f:
        pickle.dump({'words': valid_words, 'embeddings': embeddings}, f)
    logging.info(f"Filtered embeddings saved to {filtered_path}")
    
    return valid_words, embeddings

def main():
    config = load_config("configs/config.yaml")
    setup_logging()
    
    embed_config = config['embeddings']['fasttext']
    processed_data_dir = config['data_preparation']['processed_data_dir']
    embeddings_dir = config['embeddings']['filtered_embeddings_dir']
    os.makedirs(embeddings_dir, exist_ok=True)
    
    languages = ['en', 'hi']
    models = {}
    
    for lang in languages:
        input_file = os.path.join(processed_data_dir, f"preprocessed_{lang}.txt")
        output_model = config['embeddings']['fasttext'][f"model_{lang}_path"]
        model = train_fasttext_model(input_file, output_model, embed_config)
        models[lang] = model
    
    logging.info("FastText training completed for all languages.")

if __name__ == "__main__":
    main()
