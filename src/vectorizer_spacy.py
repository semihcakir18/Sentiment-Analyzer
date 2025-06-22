# In vectorizer_spacy.py

import spacy
import numpy as np

# Load the medium-sized English model that includes word vectors.
# This might take a moment the first time it's loaded.
# run "python -m spacy download en_core_web_md" in terminal
print("Loading spaCy model 'en_core_web_md'...")
nlp = spacy.load("en_core_web_md")
print("spaCy model loaded successfully.")

def vectorize_reviews(reviews):
    """
    Converts a list of cleaned text reviews into document vectors using spaCy's
    pre-trained word embeddings.

    The document vector is the average of its word vectors.
    """
    print(f"Vectorizing {len(reviews)} reviews using spaCy...")
    vectors = []
    
    # nlp.pipe is a much faster way to process multiple texts
    for doc in nlp.pipe(reviews):
        # doc.vector is the average of the word vectors in the document
        # We check if a vector exists (it might not for empty strings)
        if doc.has_vector:
            vectors.append(doc.vector)
        else:
            # For empty or out-of-vocabulary texts, add a zero vector of the same dimension
            vectors.append(np.zeros((nlp.vocab.vectors_length,)))

    print("Vectorization complete!")
    return np.array(vectors)