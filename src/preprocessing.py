import re
import string
import numpy as np


class TextPreprocessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        """Clean a single text string"""
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove HTML tags (like <br /> in movie reviews)
        text = re.sub(r"<[^>]+>", "", text)

        # Remove URLs
        text = re.sub(r"http\S+|www\S+", "", text)

        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def clean_reviews(self, reviews):
        """Clean a list of reviews"""
        print(f"Cleaning {len(reviews)} reviews...")

        cleaned_reviews = []
        for i, review in enumerate(reviews):
            cleaned = self.clean_text(review)
            cleaned_reviews.append(cleaned)

            # Show progress every 1000 reviews
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

        print("\nAFTER CLEANING:")
        print("-" * 20)
        print(cleaned[:300] + "...")

        return cleaned


# Test the preprocessing
if __name__ == "__main__":
    # Example usage
    preprocessor = TextPreprocessor()

    # Test with a sample review
    sample_text = """This movie was <br />ABSOLUTELY AMAZING!!! I can't believe how good it was. 
    The acting was phenomenal, and the plot was incredible. 10/10 would recommend! 
    Check out more reviews at www.moviesite.com"""

    print("TESTING TEXT PREPROCESSING")
    print("=" * 40)
    preprocessor.show_cleaning_example(sample_text)
