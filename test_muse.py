
from cross_align.data_loader import load_fasttext_model, get_top_n_words, load_bilingual_lexicon
from cross_align.embedding_aligner import create_embedding_matrices, procrustes_alignment, align_embeddings
from cross_align.faiss_search import build_faiss_index, translate_word
from cross_align.evaluation import evaluate_precision, compute_cosine_similarities, plot_similarity_distribution
from cross_align.utils import normalize_matrix
from cross_align.data_loader import load_bilingual_lexicon
from cross_align.faiss_search import translate_word
from cross_align.evaluation import evaluate_precision, compute_cosine_similarities, plot_similarity_distribution
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

# Define paths
muse_dir = "lexicon/"
muse_test_file = os.path.join(muse_dir, 'en-hi.test.txt')  # Make sure you have the correct path to MUSE test file

# Load MUSE test dictionary (test lexicon)
def load_muse_test_set(muse_test_file, max_pairs=None):
    """Load test set from MUSE file."""
    test_lexicon = []
    with open(muse_test_file, 'r', encoding='utf-8') as f:
        for line in f:
            source, target = line.strip().split()
            test_lexicon.append((source, target))
            if max_pairs and len(test_lexicon) >= max_pairs:
                break
    return test_lexicon

# Load test set from MUSE dataset
test_lexicon = load_muse_test_set(muse_test_file, max_pairs=10000)  # Adjust max_pairs as necessary
print(f"Loaded {len(test_lexicon)} test word pairs from MUSE dataset.")

# Precision Evaluation on the MUSE Test Set
def test_alignment_on_muse(test_lexicon, english_vocab, aligned_embeddings_en, index_hi, hindi_vocab):
    """Test the cross-lingual alignment using MUSE test set."""
    # Evaluate Precision@1 and Precision@5
    precision_at_1, precision_at_5 = evaluate_precision(test_lexicon, english_vocab, aligned_embeddings_en, index_hi, hindi_vocab)
    print(f"Precision@1 on MUSE test set: {precision_at_1:.4f}")
    print(f"Precision@5 on MUSE test set: {precision_at_5:.4f}")

    # Compute Cosine Similarities
    cosine_similarities = compute_cosine_similarities(test_lexicon, english_vocab, aligned_embeddings_en, hindi_vocab, hindi_matrix)

    # Plot similarity distribution
    plot_similarity_distribution(cosine_similarities)

# Assuming that `aligned_eng_matrix_norm` contains aligned and normalized English embeddings
# Assuming that `index_hi` is the FAISS index for Hindi embeddings
# Assuming that `hindi_vocab` contains the Hindi vocabulary from the model
# Example usage:

if __name__ == "__main__":
    # Call this function after you have performed Procrustes alignment and built the FAISS index
    test_alignment_on_muse(test_lexicon, english_vocab, aligned_eng_matrix ,index_hi, hindi_vocab)
