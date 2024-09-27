# evaluation.py
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from cross_align.alignement import align_embeddings, apply_alignment
import seaborn as sns
import tqdm
from scipy.interpolate import make_interp_spline

def word_translation_accuracy(src_emb, tgt_emb, src_words, tgt_words, test_dict, k=5):
    correct_1 = correct_5 = total = 0

    tgt_vecs = np.array([tgt_emb.get_word_vector(word) for word in tgt_words])
    
    for source, target in test_dict:
        if source in src_words and target in tgt_words:
            total += 1
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
    similarities = []
    for src_word, tgt_word in word_pairs:
        if src_word in src_words and tgt_word in tgt_words:
            sim = cosine_similarity(src_emb[src_word].reshape(1, -1), 
                                    tgt_emb[tgt_word].reshape(1, -1))[0][0]
            similarities.append((src_word, tgt_word, sim))
    return similarities

def ablation_study(src_emb, tgt_emb, src_words, tgt_words, train_dict, test_dict, sizes):
    results = []
    for size in sizes:
        train_subset = train_dict[:size]
        aligned_emb = align_embeddings(src_emb, tgt_emb, src_words, tgt_words, train_subset)
        en_aligned_supervised = apply_alignment(src_emb, aligned_emb)
        p1, p5 = word_translation_accuracy(en_aligned_supervised , tgt_emb, src_words, tgt_words, test_dict)
        results.append((size, p1, p5))
    return results

def plot_ablation_results(results, model_type):
    sizes, p1_scores, p5_scores = zip(*results)
    
    # Convert sizes and scores to numpy arrays for interpolation
    sizes = np.array(sizes)
    p1_scores = np.array(p1_scores)
    p5_scores = np.array(p5_scores)
    
    # Check the number of unique size values
    num_points = len(np.unique(sizes))
    
    # Choose the degree of the spline based on the number of points
    if num_points >= 4:
        k = 3  # Cubic spline
    elif num_points == 3:
        k = 2  # Quadratic spline
    elif num_points == 2:
        k = 1  # Linear interpolation
    else:
        raise ValueError("Not enough points for interpolation.")
    
    # Generate new size values for smooth curves (more points for a smooth line)
    size_smooth = np.linspace(sizes.min(), sizes.max(), 300)
    
    # Create smooth curves using interpolation
    p1_smooth = make_interp_spline(sizes, p1_scores, k=k)(size_smooth)
    p5_smooth = make_interp_spline(sizes, p5_scores, k=k)(size_smooth)
    
    # Plot the smooth curves
    plt.figure(figsize=(10, 6))
    plt.plot(size_smooth, p1_smooth, label='Precision@1', linewidth=2)
    plt.plot(size_smooth, p5_smooth, label='Precision@5', linewidth=2)
    
    # Customize plot appearance
    plt.xlabel('Training Dictionary Size')
    plt.ylabel('Precision')
    plt.title(f'Ablation Study {model_type} Model: Impact of Training Dictionary Size')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_similarity_distribution(similarities, model_type):
    sim_scores = [sim for _, _, sim in similarities]
    plt.figure(figsize=(10, 6))
    sns.histplot(sim_scores, bins=50, kde=True, color='skyblue')
    plt.title(f"Cosine Similarity Distribution for {model_type} model between Aligned English and Hindi Word Pairs")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Frequency")
    plt.show()
