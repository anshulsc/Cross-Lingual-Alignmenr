import fasttext
import os

def load_fasttext_model(embedding_dir, lang_code, trained=True):
    """Load the FastText model for a given language."""
    if trained:
        model_path = os.path.join(embedding_dir, f"trained/fasttext_{lang_code}.bin")
    else:
        model_path = os.path.join(embedding_dir, f"pretrained/cc.{lang_code}.300.bin")
    return fasttext.load_model(model_path)

def get_top_n_words(model, n=100000):
    """Extract the top N frequent words from a FastText model."""
    words = model.get_words()
    return words[:n]

def load_bilingual_lexicon(muse_dir, source_lang='en', target_lang='hi', max_pairs=None, train=True):
    """Load bilingual lexicon from MUSE dataset."""
    if train:
        lexicon_path = os.path.join(muse_dir, f"{source_lang}-{target_lang}.txt")
    else:
        lexicon_path = os.path.join(muse_dir, f"{source_lang}-{target_lang}.test.txt")
    lexicon = []
    with open(lexicon_path, 'r', encoding='utf-8') as f:
        for line in f:
            source, target = line.strip().split()
            lexicon.append((source, target))
            if max_pairs and len(lexicon) >= max_pairs:
                break
    return lexicon
