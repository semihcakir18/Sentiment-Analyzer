
import numpy as np
from collections import Counter
import pickle 
class SimpleTokenizer:
    """
    A simple from-scratch tokenizer to convert text to sequences of integers.
    """
    def __init__(self, num_words=None):
        """
        Initializes the tokenizer.
        :param num_words: The maximum number of words to keep in the vocabulary.
                          If None, all words are kept.
        """
        self.word_index = {}  # Dictionary to map words to integers
        self.index_word = {}  # Dictionary to map integers back to words
        self.num_words = num_words

    def fit_on_texts(self, texts):
        """
        Builds the word vocabulary from a list of texts.
        :param texts: A list of strings (e.g., ['this is a review', 'another review'])
        """
        print("Building vocabulary...")
        # Create a single list of all words from all texts
        all_words = " ".join(texts).split()
        
        # Count the frequency of each word
        word_counts = Counter(all_words)
        
        # Get the most common words, respecting the num_words limit
        if self.num_words:
            # We subtract 1 to make space for the special padding token
            most_common = word_counts.most_common(self.num_words - 1)
        else:
            most_common = word_counts.most_common()

        # Start assigning integers from 1, because 0 will be our padding token
        self.word_index['<PAD>'] = 0 
        for i, (word, count) in enumerate(most_common):
            self.word_index[word] = i + 1
        
        # Create the reverse mapping
        self.index_word = {idx: word for word, idx in self.word_index.items()}
        print(f"Vocabulary built with {len(self.word_index)} unique words.")

    def texts_to_sequences(self, texts):
        """
        Converts a list of texts into sequences of integers.
        Words not in the vocabulary will be ignored.
        :param texts: A list of strings.
        :return: A list of lists of integers.
        """
        sequences = []
        for text in texts:
            sequence = []
            for word in text.split():
                # Get the index for the word; if not found, it's ignored 
                index = self.word_index.get(word)
                if index is not None:
                    sequence.append(index)
            sequences.append(sequence)
        return sequences

    def save(self, filepath):
        """Saves the tokenizer's word_index to a file."""
        with open(filepath, 'wb') as f:
            pickle.dump(self.word_index, f)
        print(f"Tokenizer saved to {filepath}")

    def load(self, filepath):
        """Loads the tokenizer's word_index from a file."""
        with open(filepath, 'rb') as f:
            self.word_index = pickle.load(f)
        self.index_word = {idx: word for word, idx in self.word_index.items()}
        print(f"Tokenizer loaded from {filepath}")


def pad_sequences(sequences, maxlen):
    """
    Pads sequences to the same length.
    :param sequences: List of lists of integers.
    :param maxlen: The desired final length of each sequence.
    :return: A 2D NumPy array of shape (num_sequences, maxlen).
    """
    # Create a NumPy array of zeros with the correct shape
    padded = np.zeros((len(sequences), maxlen), dtype=int)
    
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        
        if len(seq) > maxlen:
            # Truncate sequences that are too long (from the beginning)
            padded[i] = seq[-maxlen:]
        else:
            # Pad sequences that are too short (at the beginning)
            padded[i, -len(seq):] = seq
            
    return padded