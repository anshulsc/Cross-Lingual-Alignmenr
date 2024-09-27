import logging
from cross_align.data_loader import load_fasttext_model, get_top_n_words, load_bilingual_lexicon
from cross_align.alignement import align_embeddings, apply_alignment, iterative_procrustes_alignment
from cross_align.evaluation import word_translation_accuracy, analyze_cosine_similarities, ablation_study, plot_ablation_results, plot_similarity_distribution
import numpy as np

def setup_logging():
    log_format = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO,
                        format=log_format,
                    handlers=[
                            logging.FileHandler("logs/alignment.log"),   
                            logging.StreamHandler()            
                        ])

def main():
    setup_logging()
    logging.info("Starting cross-lingual alignment pipeline...")
    embedding_dir = "./embeddings/"
    muse_dir = "lexicon/"
    model_type = "trained"

    logging.info(f"Loading {model_type} FastText models...")
    en_embeddings = load_fasttext_model(embedding_dir, 'en', trained=True)
    hi_embeddings = load_fasttext_model(embedding_dir, 'hi', trained=True)


    logging.info("Extracting top 100000 words from FastText models...")
    en_words = get_top_n_words(en_embeddings)
    hi_words = get_top_n_words(hi_embeddings)

    logging.info("Loading bilingual lexicon...")
    train_dict = load_bilingual_lexicon(muse_dir, 'en', 'hi')
    test_dict = load_bilingual_lexicon(muse_dir, 'en', 'hi', train=False)

    logging.info("Performing supervised alignment...")
    alignment_matrix = align_embeddings(en_embeddings, hi_embeddings, en_words, hi_words, train_dict)
    logging.info("Initial alignment matrix computed.")
    en_aligned_supervised = apply_alignment(en_embeddings, alignment_matrix)


    logging.info("Evaluating supervised alignment...")
    p1, p5 = word_translation_accuracy(en_aligned_supervised, hi_embeddings, en_words, hi_words, test_dict)
    logging.info(f"Supervised Alignment Results for {model_type} model: Precision@1: {p1:.4f}, Precision@5: {p5:.4f}")


    logging.info("Analyzing cosine similarities...")
    word_pairs = load_bilingual_lexicon(muse_dir, 'en', 'hi', max_pairs=1000, train=False)
    similarities = analyze_cosine_similarities(en_aligned_supervised, hi_embeddings, en_words, hi_words, word_pairs)
    

    sizes = [5000, 10000, 20000]
    logging.info("Starting ablation study with sizes: 5000, 10000, 20000")
    ablation_results = ablation_study(en_embeddings, hi_embeddings, en_words, hi_words, train_dict, test_dict, sizes)
    for  size, p1, p5 in ablation_results:
        logging.info(f"For sizes {size}: P@1` = {p1:.4f}, P@5 = {p5:.4f}")
    logging.info("Plotting ablation study results...")
    plot_ablation_results(ablation_results, model_type)
    plot_similarity_distribution(similarities, model_type)
    logging.info("Ablation study completed and plotted.")

if __name__ == "__main__":
    main()
