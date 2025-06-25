import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def clean_text(self, text):
        """Clean a single text string with stop word removal and lemmatization."""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"<[^>]+>", "", text)
        text = re.sub(r"http\S+|www\S+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r'\d+', '', text)

        words = text.split()

        # ### NEW and CORRECTED LOGIC ###
        # First, remove stop words, THEN lemmatize the remaining words.
        cleaned_words = []
        for word in words:
            # 1. Check if the word is a stop word
            if word not in self.stop_words:
                # 2. If it's not, THEN lemmatize it
                lemmatized_word = self.lemmatizer.lemmatize(word)
                cleaned_words.append(lemmatized_word)

        text = " ".join(cleaned_words)
        text = " ".join(text.split())

        return text

    def clean_reviews(self, reviews):
        """Clean a list of reviews"""
        print(f"Cleaning {len(reviews)} reviews...")

        cleaned_reviews = []
        for i, review in enumerate(reviews):
            cleaned = self.clean_text(review)
            cleaned_reviews.append(cleaned)

            if (i + 1) % 1000 == 0:
                print(f"Cleaned {i + 1}/{len(reviews)} reviews")

        print("Text cleaning completed!")
        return cleaned_reviews

    def show_cleaning_example(self, original_text):
        """Show before and after cleaning"""
        cleaned = self.clean_text(original_text)

        print("BEFORE CLEANING:")
        print("-" * 20)
        print(original_text[:300] + "...")

        print("\nAFTER CLEANING (Corrected Order):")
        print("-" * 20)
        print(cleaned[:300] + "...")

        return cleaned


if __name__ == "__main__":
    preprocessor = TextPreprocessor()

    sample_text = """This movie was <br />ABSOLUTELY AMAZING!!! I can't believe how good it was. 
    The actors were phenomenal, and the plot was incredible. 10/10 would recommend! 
    My friends and I are always talking about movies like this one.
    Check out more reviews at www.moviesite.com"""

    print("TESTING CORRECTED TEXT PREPROCESSING")
    print("=" * 40)
    preprocessor.show_cleaning_example(sample_text)
    