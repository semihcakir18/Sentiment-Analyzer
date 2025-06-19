import os
import numpy as np
from pathlib import Path


class IMDBDataLoader:
    def __init__(self, data_path="data/aclImdb_v1/aclImdb"):
        self.data_path = data_path

    def load_reviews_from_folder(self, folder_path):
        """Load all text files from a folder and return list of reviews"""
        reviews = []
        folder = Path(folder_path)

        if not folder.exists():
            print(f"Warning: Folder {folder_path} does not exist!")
            return reviews

        # Get all .txt files in the folder
        txt_files = list(folder.glob("*.txt"))
        print(f"Found {len(txt_files)} files in {folder_path}")

        for file_path in txt_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                    reviews.append(content)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

        return reviews

    def load_training_data(self):
        """Load training data: positive and negative reviews"""
        print("Loading training data...")

        # Load positive reviews
        pos_path = os.path.join(self.data_path, "train", "pos")
        pos_reviews = self.load_reviews_from_folder(pos_path)

        # Load negative reviews
        neg_path = os.path.join(self.data_path, "train", "neg")
        neg_reviews = self.load_reviews_from_folder(neg_path)

        # Combine reviews and create labels
        all_reviews = pos_reviews + neg_reviews
        labels = [1] * len(pos_reviews) + [0] * len(
            neg_reviews
        )  # 1=positive, 0=negative

        print(f"Loaded {len(pos_reviews)} positive reviews")
        print(f"Loaded {len(neg_reviews)} negative reviews")
        print(f"Total: {len(all_reviews)} reviews")

        return all_reviews, np.array(labels)

    def load_test_data(self):
        """Load test data: positive and negative reviews"""
        print("Loading test data...")

        # Load positive reviews
        pos_path = os.path.join(self.data_path, "test", "pos")
        pos_reviews = self.load_reviews_from_folder(pos_path)

        # Load negative reviews
        neg_path = os.path.join(self.data_path, "test", "neg")
        neg_reviews = self.load_reviews_from_folder(neg_path)

        # Combine reviews and create labels
        all_reviews = pos_reviews + neg_reviews
        labels = [1] * len(pos_reviews) + [0] * len(neg_reviews)

        print(f"Loaded {len(pos_reviews)} positive test reviews")
        print(f"Loaded {len(neg_reviews)} negative test reviews")
        print(f"Total: {len(all_reviews)} test reviews")

        return all_reviews, np.array(labels)


# Example usage function
def explore_data():
    """Function to explore what the data looks like"""
    loader = IMDBDataLoader()

    # Load training data
    reviews, labels = loader.load_training_data()

    if len(reviews) > 0:
        print("\n" + "=" * 50)
        print("DATA EXPLORATION")
        print("=" * 50)

        # Show first positive review
        first_positive_idx = np.where(labels == 1)[0][0]
        print(f"\nFIRST POSITIVE REVIEW (label={labels[first_positive_idx]}):")
        print("-" * 30)
        print(reviews[first_positive_idx][:500] + "...")  # First 500 characters

        # Show first negative review
        first_negative_idx = np.where(labels == 0)[0][0]
        print(f"\nFIRST NEGATIVE REVIEW (label={labels[first_negative_idx]}):")
        print("-" * 30)
        print(reviews[first_negative_idx][:500] + "...")

        # Show some statistics
        print(f"\nDATASET STATISTICS:")
        print(f"Total reviews: {len(reviews)}")
        print(f"Positive reviews: {np.sum(labels == 1)}")
        print(f"Negative reviews: {np.sum(labels == 0)}")
        print(
            f"Average review length: {np.mean([len(review) for review in reviews]):.0f} characters"
        )

    return reviews, labels


if __name__ == "__main__":
    # This runs when you execute: python src/data_loader.py
    explore_data()
