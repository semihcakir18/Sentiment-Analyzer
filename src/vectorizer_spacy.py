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
    Converts a list of RAW text reviews into document vectors using spaCy.
    """
    print(f"Vectorizing {len(reviews)} reviews using spaCy...")
    vectors = []

    for doc in nlp.pipe(reviews, disable=["parser", "ner"]):
        vectors.append(doc.vector)

    print("Vectorization complete!")
    return np.array(vectors)
