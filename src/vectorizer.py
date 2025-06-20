import numpy as np
from collections import Counter
import pickle

class BagOfWordsVectorizer:
    def __init__(self, max_features=5000, min_word_freq=5):
        """
        max_features: Maximum number of words to keep in vocabulary
        min_word_freq: Minimum frequency for a word to be included
        """
        self.max_features = max_features
        self.min_word_freq = min_word_freq
        self.word_to_index = {}
        self.index_to_word = {}
        self.vocabulary_size = 0
        
    def fit(self, texts):
        """Learn vocabulary from training texts"""
        print(f"Building vocabulary from {len(texts)} texts...")
        
        # Count all words across all texts
        word_counts = Counter()
        for text in texts:
            words = text.split()
            word_counts.update(words)
        
        print(f"Found {len(word_counts)} unique words")
        
        # Filter words by minimum frequency
        filtered_words = {word: count for word, count in word_counts.items() 
                         if count >= self.min_word_freq}
        
        print(f"After filtering (min_freq={self.min_word_freq}): {len(filtered_words)} words")
        
        # Take most common words up to max_features
        most_common = Counter(filtered_words).most_common(self.max_features)
        
        # Create word-to-index mapping
        for i, (word, count) in enumerate(most_common):
            self.word_to_index[word] = i
            self.index_to_word[i] = word
            
        self.vocabulary_size = len(self.word_to_index)
        print(f"Final vocabulary size: {self.vocabulary_size}")
        
        # Show some example words
        print(f"Most common words: {list(self.word_to_index.keys())[:10]}")
        
    def transform(self, texts):
        """Convert texts to bag-of-words vectors"""
        print(f"Vectorizing {len(texts)} texts...")
        
        vectors = []
        for i, text in enumerate(texts):
            # Create zero vector
            vector = np.zeros(self.vocabulary_size)
            
            # Count words in this text
            words = text.split()
            for word in words:
                if word in self.word_to_index:
                    word_idx = self.word_to_index[word]
                    vector[word_idx] += 1
            
            vectors.append(vector)
            
            # Show progress
            if (i + 1) % 1000 == 0:
                print(f"Vectorized {i + 1}/{len(texts)} texts")
        
        return np.array(vectors)
    
    def fit_transform(self, texts):
        """Fit vocabulary and transform texts in one step"""
        self.fit(texts)
        return self.transform(texts)
    
    def get_feature_names(self):
        """Get list of words in vocabulary"""
        return [self.index_to_word[i] for i in range(self.vocabulary_size)]
    
    def save_vocabulary(self, filepath):
        """Save vocabulary to file"""
        vocab_data = {
            'word_to_index': self.word_to_index,
            'index_to_word': self.index_to_word,
            'vocabulary_size': self.vocabulary_size,
            'max_features': self.max_features,
            'min_word_freq': self.min_word_freq
        }
        with open(filepath, 'wb') as f:
            pickle.dump(vocab_data, f)
        print(f"Vocabulary saved to {filepath}")
    
    def load_vocabulary(self, filepath):
        """Load vocabulary from file"""
        with open(filepath, 'rb') as f:
            vocab_data = pickle.load(f)
        
        self.word_to_index = vocab_data['word_to_index']
        self.index_to_word = vocab_data['index_to_word']
        self.vocabulary_size = vocab_data['vocabulary_size']
        self.max_features = vocab_data['max_features']
        self.min_word_freq = vocab_data['min_word_freq']
        print(f"Vocabulary loaded from {filepath}")

# Test the vectorizer
def test_vectorizer():
    """Test the bag of words vectorizer"""
    print("🧪 TESTING BAG OF WORDS VECTORIZER")
    print("=" * 40)
    
    # Sample texts
    sample_texts = [
        "this movie was great i loved it",
        "terrible movie very bad acting",
        "amazing film great acting loved it",
        "worst movie ever very bad"
    ]
    
    # Create and fit vectorizer
    vectorizer = BagOfWordsVectorizer(max_features=20, min_word_freq=1)
    vectors = vectorizer.fit_transform(sample_texts)
    
    print(f"\nVectorizer results:")
    print(f"Vocabulary: {vectorizer.get_feature_names()}")
    print(f"Vector shape: {vectors.shape}")
    
    # Show first vector
    print(f"\nFirst text: '{sample_texts[0]}'")
    print(f"First vector: {vectors[0]}")
    
    # Show word counts for first text
    print(f"\nWord counts in first text:")
    for word_idx, count in enumerate(vectors[0]):
        if count > 0:
            word = vectorizer.index_to_word[word_idx]
            print(f"  '{word}': {count}")

if __name__ == "__main__":
    test_vectorizer()