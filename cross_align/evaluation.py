# evaluation.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from cross_align.alignement import align_embeddings, apply_alignment
import seaborn as sns
import tqdm

def word_translation_accuracy(src_emb, tgt_emb, src_words, tgt_words, test_dict, k=5):
    """Evaluate word translation accuracy."""
    correct_1 = correct_5 = total = 0
    tgt_vecs = np.array([tgt_emb.get_word_vector(word) for word in tgt_words])
    for source, target in test_dict:
        if source in src_words and target in tgt_words:
            total += 1
            if total % 1000 == 0:
                print(f"Processed {total} word pairs")
            src_vec = src_emb[source].reshape(1, -1)
            similarities = cosine_similarity(src_vec,  tgt_vecs)[0]
            top_k = np.argsort(similarities)[-k:][::-1]
            
            if tgt_words[top_k[0]] == target:
                correct_1 += 1
            if target in [tgt_words[idx] for idx in top_k]:
                correct_5 += 1
    p1 = correct_1 / total
    p5 = correct_5 / total
    return p1, p5

def analyze_cosine_similarities(src_emb, tgt_emb, src_words, tgt_words, word_pairs):
    """Compute and analyze cosine similarities between word pairs."""
    similarities = []
    tgt_vecs = np.array([tgt_emb.get_word_vector(word) for word in tgt_words])
    for src_word, tgt_word in word_pairs:
        if src_word in src_words and tgt_word in tgt_words:
            sim = cosine_similarity(src_emb[src_word].reshape(1, -1), 
                                    tgt_emb[tgt_word].reshape(1, -1))[0][0]
            similarities.append((src_word, tgt_word, sim))
    return similarities

def ablation_study(src_emb, tgt_emb, src_words, tgt_words, train_dict, test_dict, sizes):
    """Perform ablation study with different training dictionary sizes."""
    results = []
    for size in sizes:
        train_subset = train_dict[:size]
        aligned_emb = align_embeddings(src_emb, tgt_emb, src_words, tgt_words, train_subset)
        en_aligned_supervised = apply_alignment(src_emb, aligned_emb)
        p1, p5 = word_translation_accuracy(en_aligned_supervised , tgt_emb, src_words, tgt_words, test_dict)
        results.append((size, p1, p5))
    return results

def plot_ablation_results(results):
    """Plot ablation study results."""

    sizes, p1_scores, p5_scores = zip(*results)
    plt.figure(figsize=(10, 6))
    plt.plot(sizes, p1_scores, marker='o', label='Precision@1')
    plt.plot(sizes, p5_scores, marker='o', label='Precision@5')
    plt.xlabel('Training Dictionary Size')
    plt.ylabel('Precision')
    plt.title('Ablation Study: Impact of Training Dictionary Size')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_similarity_distribution(similarities):
    """
    Plots the distribution of cosine similarity scores.

    Parameters:
    - similarities: List of cosine similarity scores.
    """
    sim_scores = [sim for _, _, sim in similarities]
    
    plt.figure(figsize=(10, 6))
    sns.histplot(sim_scores, bins=50, kde=True, color='skyblue')
    plt.title("Cosine Similarity Distribution between Aligned English and Hindi Word Pairs")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.show()
