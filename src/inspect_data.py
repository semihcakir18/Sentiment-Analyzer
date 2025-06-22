import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import Counter

# Add src to path
sys.path.append(os.path.dirname(__file__))

from data_loader import IMDBDataLoader
from preprocessing import TextPreprocessor
from vectorizer import BagOfWordsVectorizer

def inspect_raw_data(reviews, labels, n_samples=5):
    """Inspect raw data before any processing"""
    print("🔍 RAW DATA INSPECTION")
    print("=" * 50)
    
    print(f"Total samples: {len(reviews)}")
    print(f"Positive samples: {np.sum(labels == 1)}")
    print(f"Negative samples: {np.sum(labels == 0)}")
    print(f"Class balance: {np.mean(labels):.3f}")
    
    # Sample random reviews
    indices = random.sample(range(len(reviews)), n_samples)
    
    for i, idx in enumerate(indices):
        sentiment = "POSITIVE" if labels[idx] == 1 else "NEGATIVE"
        print(f"\n--- Sample {i+1} ({sentiment}) ---")
        print(f"Length: {len(reviews[idx])} characters")
        print(f"Preview: {reviews[idx][:200]}...")
        if len(reviews[idx]) > 200:
            print("...")

def inspect_cleaned_data(original_reviews, cleaned_reviews, labels, n_samples=5):
    """Compare original vs cleaned data"""
    print("\n🧹 CLEANED DATA INSPECTION")
    print("=" * 50)
    
    # Calculate cleaning statistics
    original_lengths = [len(review) for review in original_reviews]
    cleaned_lengths = [len(review) for review in cleaned_reviews]
    
    print(f"Average original length: {np.mean(original_lengths):.1f} chars")
    print(f"Average cleaned length: {np.mean(cleaned_lengths):.1f} chars")
    print(f"Average reduction: {(1 - np.mean(cleaned_lengths)/np.mean(original_lengths))*100:.1f}%")
    
    # Sample comparisons
    indices = random.sample(range(len(original_reviews)), n_samples)
    
    for i, idx in enumerate(indices):
        sentiment = "POSITIVE" if labels[idx] == 1 else "NEGATIVE"
        print(f"\n--- Sample {i+1} ({sentiment}) ---")
        print(f"ORIGINAL ({len(original_reviews[idx])} chars):")
        print(f"{original_reviews[idx][:150]}...")
        print(f"\nCLEANED ({len(cleaned_reviews[idx])} chars):")
        print(f"{cleaned_reviews[idx][:150]}...")
        
        # Word count comparison
        orig_words = len(original_reviews[idx].split())
        clean_words = len(cleaned_reviews[idx].split())
        print(f"Words: {orig_words} → {clean_words}")

def inspect_vocabulary(vectorizer, top_n=20):
    """Inspect the vocabulary created by vectorizer"""
    print("\n📚 VOCABULARY INSPECTION")
    print("=" * 50)
    
    print(f"Vocabulary size: {vectorizer.vocabulary_size}")
    
    if hasattr(vectorizer, 'word_counts'):
        # Get most common words
        most_common = vectorizer.word_counts.most_common(top_n)
        print(f"\nTop {top_n} most frequent words:")
        for word, count in most_common:
            print(f"  {word}: {count}")
        
        # Get least common words (that made it into vocabulary)
        least_common = vectorizer.word_counts.most_common()[-top_n:]
        print(f"\nTop {top_n} least frequent words (in vocabulary):")
        for word, count in reversed(least_common):
            print(f"  {word}: {count}")

def visualize_vectors_2d(X, labels, n_samples=1000, method='pca'):
    """Visualize high-dimensional vectors in 2D"""
    print(f"\n🗺️  VECTOR VISUALIZATION ({method.upper()})")
    print("=" * 50)
    
    # Sample data if too large
    if len(X) > n_samples:
        indices = random.sample(range(len(X)), n_samples)
        X_sample = X[indices]
        labels_sample = labels[indices]
        print(f"Sampling {n_samples} points from {len(X)} total")
    else:
        X_sample = X
        labels_sample = labels
    
    if method == 'pca':
        # Simple PCA implementation
        X_centered = X_sample - np.mean(X_sample, axis=0)
        cov_matrix = np.cov(X_centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Sort by eigenvalues (descending)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        
        # Project to 2D
        X_2d = X_centered @ eigenvectors[:, :2]
        
        print(f"Explained variance ratio: {eigenvalues[idx[:2]]/np.sum(eigenvalues)}")
        
    elif method == 'random':
        # Random projection (much faster)
        random_matrix = np.random.randn(X_sample.shape[1], 2)
        random_matrix = random_matrix / np.linalg.norm(random_matrix, axis=0)
        X_2d = X_sample @ random_matrix
        
    else:
        raise ValueError("Method must be 'pca' or 'random'")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot positive and negative samples with different colors
    pos_mask = labels_sample == 1
    neg_mask = labels_sample == 0
    
    plt.scatter(X_2d[pos_mask, 0], X_2d[pos_mask, 1], 
               c='red', alpha=0.6, label=f'Positive ({np.sum(pos_mask)})', s=20)
    plt.scatter(X_2d[neg_mask, 0], X_2d[neg_mask, 1], 
               c='blue', alpha=0.6, label=f'Negative ({np.sum(neg_mask)})', s=20)
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'Sentiment Vectors Visualization ({method.upper()})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add some statistics
    pos_center = np.mean(X_2d[pos_mask], axis=0)
    neg_center = np.mean(X_2d[neg_mask], axis=0)
    
    plt.scatter(*pos_center, c='darkred', s=100, marker='x', linewidth=3, label='Positive Center')
    plt.scatter(*neg_center, c='darkblue', s=100, marker='x', linewidth=3, label='Negative Center')
    
    distance = np.linalg.norm(pos_center - neg_center)
    print(f"Distance between class centers: {distance:.3f}")
    
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_vector_statistics(X, labels):
    """Analyze statistical properties of vectors"""
    print("\n📊 VECTOR STATISTICS")
    print("=" * 50)
    
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    X_pos = X[pos_mask]
    X_neg = X[neg_mask]
    
    print(f"Vector dimensions: {X.shape[1]}")
    print(f"Positive samples: {X_pos.shape[0]}")
    print(f"Negative samples: {X_neg.shape[0]}")
    
    # Sparsity analysis
    sparsity_all = np.mean(X == 0)
    sparsity_pos = np.mean(X_pos == 0)
    sparsity_neg = np.mean(X_neg == 0)
    
    print(f"\nSparsity (% of zeros):")
    print(f"  Overall: {sparsity_all:.3f}")
    print(f"  Positive: {sparsity_pos:.3f}")
    print(f"  Negative: {sparsity_neg:.3f}")
    
    # Magnitude analysis
    norms_all = np.linalg.norm(X, axis=1)
    norms_pos = np.linalg.norm(X_pos, axis=1)
    norms_neg = np.linalg.norm(X_neg, axis=1)
    
    print(f"\nVector magnitudes:")
    print(f"  Overall: mean={np.mean(norms_all):.3f}, std={np.std(norms_all):.3f}")
    print(f"  Positive: mean={np.mean(norms_pos):.3f}, std={np.std(norms_pos):.3f}")
    print(f"  Negative: mean={np.mean(norms_neg):.3f}, std={np.std(norms_neg):.3f}")
    
    # Feature usage analysis
    feature_usage = np.mean(X > 0, axis=0)
    print(f"\nFeature usage:")
    print(f"  Most used features: {np.max(feature_usage):.3f}")
    print(f"  Least used features: {np.min(feature_usage):.3f}")
    print(f"  Average feature usage: {np.mean(feature_usage):.3f}")

def plot_vector_distributions(X, labels):
    """Plot distributions of vector properties"""
    print("\n📈 VECTOR DISTRIBUTIONS")
    print("=" * 50)
    
    pos_mask = labels == 1
    neg_mask = labels == 0
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Vector magnitudes
    norms_pos = np.linalg.norm(X[pos_mask], axis=1)
    norms_neg = np.linalg.norm(X[neg_mask], axis=1)
    
    axes[0, 0].hist(norms_pos, bins=50, alpha=0.7, label='Positive', color='red')
    axes[0, 0].hist(norms_neg, bins=50, alpha=0.7, label='Negative', color='blue')
    axes[0, 0].set_xlabel('Vector Magnitude')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Vector Magnitudes')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Number of non-zero features
    nonzero_pos = np.sum(X[pos_mask] > 0, axis=1)
    nonzero_neg = np.sum(X[neg_mask] > 0, axis=1)
    
    axes[0, 1].hist(nonzero_pos, bins=50, alpha=0.7, label='Positive', color='red')
    axes[0, 1].hist(nonzero_neg, bins=50, alpha=0.7, label='Negative', color='blue')
    axes[0, 1].set_xlabel('Number of Non-zero Features')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Active Features')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature values distribution
    pos_values = X[pos_mask][X[pos_mask] > 0]
    neg_values = X[neg_mask][X[neg_mask] > 0]
    
    axes[1, 0].hist(pos_values, bins=50, alpha=0.7, label='Positive', color='red')
    axes[1, 0].hist(neg_values, bins=50, alpha=0.7, label='Negative', color='blue')
    axes[1, 0].set_xlabel('Feature Values (non-zero)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Non-zero Feature Values')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Average feature values per class
    avg_pos = np.mean(X[pos_mask], axis=0)
    avg_neg = np.mean(X[neg_mask], axis=0)
    
    # Show top features that differ between classes
    diff = avg_pos - avg_neg
    top_indices = np.argsort(np.abs(diff))[-100:]  # Top 100 different features
    
    axes[1, 1].scatter(avg_pos[top_indices], avg_neg[top_indices], alpha=0.6)
    axes[1, 1].plot([0, np.max([avg_pos.max(), avg_neg.max()])], 
                    [0, np.max([avg_pos.max(), avg_neg.max()])], 'r--', alpha=0.5)
    axes[1, 1].set_xlabel('Average Feature Value (Positive)')
    axes[1, 1].set_ylabel('Average Feature Value (Negative)')
    axes[1, 1].set_title('Feature Values: Positive vs Negative')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main inspection pipeline"""
    print("🔍 DATA INSPECTION PIPELINE")
    print("=" * 60)
    
    # Load data
    print("Loading data...")
    loader = IMDBDataLoader()
    reviews, labels = loader.load_training_data()
    
    if len(reviews) == 0:
        print("❌ No data loaded! Make sure dataset is in data/aclImdb/")
        return
    
    # 1. Inspect raw data
    inspect_raw_data(reviews, labels, n_samples=3)
    
    # 2. Clean data and inspect
    print("\nCleaning data...")
    preprocessor = TextPreprocessor()
    cleaned_reviews = preprocessor.clean_reviews(reviews)
    inspect_cleaned_data(reviews, cleaned_reviews, labels, n_samples=3)
    
    # 3. Vectorize and inspect
    print("\nVectorizing data...")
    vectorizer = BagOfWordsVectorizer(max_features=5000, min_word_freq=5)
    X = vectorizer.fit_transform(cleaned_reviews)
    
    inspect_vocabulary(vectorizer)
    analyze_vector_statistics(X, labels)
    
    # 4. Visualize vectors
    print("\nCreating visualizations...")
    
    # PCA visualization
    try:
        visualize_vectors_2d(X, labels, n_samples=25000, method='pca')
    except Exception as e:
        print(f"PCA visualization failed: {e}")
        print("Trying random projection...")
        # 4. Visualize vectors (continued)
    print("\nCreating visualizations...")
    
    # PCA visualization
    try:
        visualize_vectors_2d(X, labels, n_samples=25000, method='pca')
    except Exception as e:
        print(f"PCA visualization failed: {e}")
        print("Trying random projection...")
        try:
            visualize_vectors_2d(X, labels, n_samples=25000, method='random')
        except Exception as e:
            print(f"Random projection also failed: {e}")
    
    # 5. Plot distributions
    try:
        plot_vector_distributions(X, labels)
    except Exception as e:
        print(f"Distribution plotting failed: {e}")
    
    # 6. Sample some vectorized examples
    print("\n🔢 VECTORIZED EXAMPLES")
    print("=" * 50)
    
    sample_indices = random.sample(range(len(cleaned_reviews)), 3)
    for i, idx in enumerate(sample_indices):
        sentiment = "POSITIVE" if labels[idx] == 1 else "NEGATIVE"
        print(f"\n--- Vectorized Sample {i+1} ({sentiment}) ---")
        print(f"Original text: {cleaned_reviews[idx][:100]}...")
        
        # Get the vector for this sample
        sample_vector = X[idx].toarray().flatten() if hasattr(X[idx], 'toarray') else X[idx]
        
        # Find non-zero features
        nonzero_indices = np.where(sample_vector > 0)[0]
        nonzero_values = sample_vector[nonzero_indices]
        
        print(f"Vector shape: {sample_vector.shape}")
        print(f"Non-zero features: {len(nonzero_indices)}")
        print(f"Vector magnitude: {np.linalg.norm(sample_vector):.3f}")
        
        # Show top features if we have vocabulary
        if hasattr(vectorizer, 'vocab_to_index'):
            # Reverse the vocabulary mapping
            index_to_vocab = {v: k for k, v in vectorizer.vocab_to_index.items()}
            
            # Get top 10 features for this sample
            top_feature_indices = nonzero_indices[np.argsort(nonzero_values)[-10:]]
            print("Top 10 features:")
            for feat_idx in reversed(top_feature_indices):
                word = index_to_vocab.get(feat_idx, f"feature_{feat_idx}")
                value = sample_vector[feat_idx]
                print(f"  {word}: {value:.3f}")
    
    print("\n✅ Data inspection completed!")

def quick_inspection():
    """Quick inspection with minimal samples for fast debugging"""
    print("🚀 QUICK DATA INSPECTION")
    print("=" * 40)
    
    # Load small sample
    loader = IMDBDataLoader()
    reviews, labels = loader.load_training_data()
    
    if len(reviews) == 0:
        print("❌ No data found!")
        return
    
    # Take only first 1000 samples for speed
    reviews = reviews[:1000]
    labels = labels[:1000]
    
    print(f"Using {len(reviews)} samples for quick inspection")
    
    # Quick preprocessing
    preprocessor = TextPreprocessor()
    cleaned_reviews = preprocessor.clean_reviews(reviews)
    
    # Quick vectorization
    vectorizer = BagOfWordsVectorizer(max_features=1000, min_word_freq=2)
    X = vectorizer.fit_transform(cleaned_reviews)
    
    # Quick stats
    print(f"Vocabulary size: {vectorizer.vocabulary_size}")
    print(f"Vector shape: {X.shape}")
    print(f"Sparsity: {np.mean(X == 0):.3f}")
    
    # Quick visualization
    try:
        visualize_vectors_2d(X, labels, n_samples=500, method='random')
    except Exception as e:
        print(f"Visualization failed: {e}")
    
    print("✅ Quick inspection done!")

def inspect_specific_words(vectorizer, words_to_check):
    """Check if specific words are in vocabulary and their frequencies"""
    print(f"\n🔍 SPECIFIC WORD INSPECTION")
    print("=" * 50)
    
    if not hasattr(vectorizer, 'vocab_to_index'):
        print("Vectorizer doesn't have vocabulary mapping!")
        return
    
    print(f"Checking {len(words_to_check)} words...")
    
    found_words = []
    missing_words = []
    
    for word in words_to_check:
        if word in vectorizer.vocab_to_index:
            idx = vectorizer.vocab_to_index[word]
            freq = vectorizer.word_counts.get(word, 0) if hasattr(vectorizer, 'word_counts') else 'unknown'
            found_words.append((word, idx, freq))
            print(f"✅ '{word}' -> index {idx}, frequency: {freq}")
        else:
            missing_words.append(word)
            print(f"❌ '{word}' not in vocabulary")
    
    print(f"\nSummary: {len(found_words)} found, {len(missing_words)} missing")
    
    if missing_words:
        print(f"Missing words: {missing_words}")

def compare_preprocessing_methods():
    """Compare different preprocessing approaches"""
    print("\n🔄 PREPROCESSING COMPARISON")
    print("=" * 50)
    
    # Load small sample
    loader = IMDBDataLoader()
    reviews, labels = loader.load_training_data()
    sample_reviews = reviews[:100]  # Small sample for comparison
    
    preprocessor = TextPreprocessor()
    
    print("Comparing preprocessing on 100 samples...")
    
    # Method 1: Current preprocessing
    cleaned_1 = preprocessor.clean_reviews(sample_reviews)
    
    # Method 2: Minimal preprocessing (just lowercase)
    cleaned_2 = [review.lower() for review in sample_reviews]
    
    # Method 3: Aggressive preprocessing (remove more)
    def aggressive_clean(text):
        import re
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Only letters
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    cleaned_3 = [aggressive_clean(review) for review in sample_reviews]
    
    # Compare vocabulary sizes
    vectorizer1 = BagOfWordsVectorizer(max_features=1000, min_word_freq=1)
    vectorizer2 = BagOfWordsVectorizer(max_features=1000, min_word_freq=1)
    vectorizer3 = BagOfWordsVectorizer(max_features=1000, min_word_freq=1)
    
    X1 = vectorizer1.fit_transform(cleaned_1)
    X2 = vectorizer2.fit_transform(cleaned_2)
    X3 = vectorizer3.fit_transform(cleaned_3)
    
    print(f"Current preprocessing: {vectorizer1.vocabulary_size} unique words")
    print(f"Minimal preprocessing: {vectorizer2.vocabulary_size} unique words")
    print(f"Aggressive preprocessing: {vectorizer3.vocabulary_size} unique words")
    
    # Show example
    idx = 0
    print(f"\nExample comparison (sample {idx}):")
    print(f"Original: {sample_reviews[idx][:100]}...")
    print(f"Current:  {cleaned_1[idx][:100]}...")
    print(f"Minimal:  {cleaned_2[idx][:100]}...")
    print(f"Aggressive: {cleaned_3[idx][:100]}...")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect sentiment analysis data')
    parser.add_argument('--quick', action='store_true', help='Run quick inspection only')
    parser.add_argument('--words', nargs='+', help='Check specific words in vocabulary')
    parser.add_argument('--compare-preprocessing', action='store_true', help='Compare preprocessing methods')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_inspection()
    elif args.words:
        # Need to load and vectorize first
        loader = IMDBDataLoader()
        reviews, labels = loader.load_training_data()
        preprocessor = TextPreprocessor()
        cleaned_reviews = preprocessor.clean_reviews(reviews[:1000])  # Sample for speed
        vectorizer = BagOfWordsVectorizer(max_features=5000, min_word_freq=5)
        vectorizer.fit_transform(cleaned_reviews)
        inspect_specific_words(vectorizer, args.words)
    elif args.compare_preprocessing:
        compare_preprocessing_methods()
    else:
        main()
