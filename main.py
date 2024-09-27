# main.py

from cross_align.data_loader import load_fasttext_model, get_top_n_words, load_bilingual_lexicon
from cross_align.embedding_aligner import create_embedding_matrices, procrustes_alignment, align_embeddings
from cross_align.faiss_search import build_faiss_index, translate_word
from cross_align.evaluation import evaluate_precision, compute_cosine_similarities, plot_similarity_distribution
from cross_align.utils import normalize_matrix
import os
import numpy as np

# Define paths
embedding_dir = "./embeddings/pretrained/"
muse_dir = "lexicon/"

# Load FastText models and vocabs
english_model = load_fasttext_model(embedding_dir, 'en')
hindi_model = load_fasttext_model(embedding_dir, 'hi')

english_vocab = get_top_n_words(english_model)
hindi_vocab = get_top_n_words(hindi_model)

# Load bilingual lexicon
bilingual_lexicon = load_bilingual_lexicon(muse_dir, max_pairs=20000)

# Create embedding matrices
X, Y, valid_pairs = create_embedding_matrices(bilingual_lexicon, english_model, hindi_model, english_vocab, hindi_vocab)

# Procrustes alignment
W = procrustes_alignment(X, Y)
aligned_embeddings_en = align_embeddings(english_model, english_vocab, W)

# Normalize aligned and Hindi embeddings
aligned_eng_matrix = normalize_matrix(np.array([aligned_embeddings_en[word] for word in english_vocab]))
hindi_matrix = normalize_matrix(np.array([hindi_model.get_word_vector(word) for word in hindi_vocab]))

# Build FAISS index for Hindi embeddings
index_hi = build_faiss_index(hindi_matrix)

# Evaluate precision
test_lexicon = bilingual_lexicon[:10000]  # Example test set
precision_at_1, precision_at_5 = evaluate_precision(test_lexicon, english_vocab, aligned_eng_matrix, index_hi, hindi_vocab)
print(f"Precision@1: {precision_at_1:.4f}")
print(f"Precision@5: {precision_at_5:.4f}")

# Compute and plot cosine similarities
cosine_similarities = compute_cosine_similarities(test_lexicon, english_vocab, aligned_eng_matrix, hindi_vocab, hindi_matrix)
plot_similarity_distribution(cosine_similarities)
