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

from vectorizer_spacy import vectorize_reviews


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
    
    # 2. Number of non-zero features (less useful for dense vectors, but won't break)
    nonzero_pos = np.sum(X[pos_mask] != 0, axis=1)
    nonzero_neg = np.sum(X[neg_mask] != 0, axis=1)
    
    axes[0, 1].hist(nonzero_pos, bins=50, alpha=0.7, label='Positive', color='red')
    axes[0, 1].hist(nonzero_neg, bins=50, alpha=0.7, label='Negative', color='blue')
    axes[0, 1].set_xlabel('Number of Non-zero Features')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Distribution of Active Features')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Feature values distribution
    pos_values = X[pos_mask][X[pos_mask] != 0]
    neg_values = X[neg_mask][X[neg_mask] != 0]
    
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
    
    # No need to select indices, just plot all 300 dimensions
    axes[1, 1].scatter(avg_pos, avg_neg, alpha=0.6)
    min_val = min(avg_pos.min(), avg_neg.min())
    max_val = max(avg_pos.max(), avg_neg.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.5)
    axes[1, 1].set_xlabel('Average Feature Value (Positive)')
    axes[1, 1].set_ylabel('Average Feature Value (Negative)')
    axes[1, 1].set_title('Feature Values: Positive vs Negative')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main inspection pipeline with spaCy Embeddings"""
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
    
    # 3. ### FIXED ### Vectorize using the spaCy function
    print("\nVectorizing data using spaCy...")
    X = vectorize_reviews(cleaned_reviews)
    
    # ### FIXED ### The following functions are no longer relevant for embeddings
    # inspect_vocabulary(vectorizer) 
    
    # The statistical analysis will still work and provide insights on the new vectors
    analyze_vector_statistics(X, labels)
    
    # 4. Visualize vectors
    print("\nCreating visualizations...")
    try:
        visualize_vectors_2d(X, labels, n_samples=25000, method='pca')
    except Exception as e:
        print(f"PCA visualization failed: {e}")
    
    # 5. Plot distributions
    try:
        plot_vector_distributions(X, labels)
    except Exception as e:
        print(f"Distribution plotting failed: {e}")
    
    # ### FIXED ### This section is removed as it relied on a word-based vocabulary
    # print("\n🔢 VECTORIZED EXAMPLES")
    
    print("\n✅ Data inspection completed!")


# ### FIXED ### Simplified the main execution block
if __name__ == "__main__":
    main()