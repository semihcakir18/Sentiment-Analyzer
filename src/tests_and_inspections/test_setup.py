"""
This script tests if everything is set up correctly

"""

import numpy as np
import sys
import os
from pathlib import Path


from ..data_loader import IMDBDataLoader
from ..preprocessing import TextPreprocessor


def test_data_loading():
    """Test if we can load the IMDB dataset"""
    print("🧪 TESTING DATA LOADING...")
    print("=" * 40)

    try:
        loader = IMDBDataLoader()
        reviews, labels = loader.load_training_data()

        if len(reviews) == 0:
            print("❌ ERROR: No reviews loaded!")
            print(
                "   Make sure you downloaded and extracted the IMDB dataset to data/aclImdb/"
            )
            return False
        elif len(reviews) < 25000:
            print(f"⚠️  WARNING: Only loaded {len(reviews)} reviews, expected 25000")
            print("   Some files might be missing")
        else:
            print(f"✅ SUCCESS: Loaded {len(reviews)} reviews")

        print(f"   Positive reviews: {sum(labels)}")
        print(f"   Negative reviews: {len(labels) - sum(labels)}")
        return True

    except Exception as e:
        print(f"❌ ERROR loading data: {e}")
        return False


def test_preprocessing():
    """Test if text preprocessing works"""
    print("\n🧪 TESTING TEXT PREPROCESSING...")
    print("=" * 40)

    try:
        preprocessor = TextPreprocessor()

        # Test with sample text
        sample = "This movie was <br />GREAT!!! I loved it. 10/10 stars!"
        cleaned = preprocessor.clean_text(sample)

        print(f"Original: {sample}")
        print(f"Cleaned:  {cleaned}")

        if len(cleaned) > 0 and "<br />" not in cleaned and "!!!" not in cleaned:
            print("✅ SUCCESS: Text preprocessing works!")
            return True
        else:
            print("❌ ERROR: Text preprocessing not working properly")
            return False

    except Exception as e:
        print(f"❌ ERROR in preprocessing: {e}")
        return False


def test_full_pipeline():
    """Test loading data and preprocessing together"""
    print("\n🧪 TESTING FULL PIPELINE...")
    print("=" * 40)

    try:
        # Load a small sample of data
        loader = IMDBDataLoader()
        reviews, labels = loader.load_training_data()

        if len(reviews) == 0:
            print("❌ Cannot test pipeline - no data loaded")
            return False

        # Take first 10 reviews for testing
        sample_reviews = reviews[:10]
        sample_labels = labels[:10]

        # Preprocess them
        preprocessor = TextPreprocessor()
        cleaned_reviews = preprocessor.clean_reviews(sample_reviews)

        print(f"✅ SUCCESS: Processed {len(cleaned_reviews)} sample reviews")
        print(
            f"   Average length before: {np.mean([len(r) for r in sample_reviews]):.0f} chars"
        )
        print(
            f"   Average length after:  {np.mean([len(r) for r in cleaned_reviews]):.0f} chars"
        )

        return True

    except Exception as e:
        print(f"❌ ERROR in full pipeline: {e}")
        return False


def main():
    """Run all tests"""
    print("🚀 SETUP TEST")
    print("=" * 50)

    # Check if numpy is available
    try:
        import numpy as np

        print("✅ NumPy is installed")
    except ImportError:
        print("❌ NumPy is not installed. Run: pip install numpy")
        return

    # Run tests
    data_test = test_data_loading()
    preprocessing_test = test_preprocessing()

    if data_test:
        pipeline_test = test_full_pipeline()
    else:
        pipeline_test = False

    # Final results
    print("\n" + "=" * 50)
    print("📊 TEST RESULTS SUMMARY")
    print("=" * 50)
    print(f"Data Loading:     {'✅ PASS' if data_test else '❌ FAIL'}")
    print(f"Preprocessing:    {'✅ PASS' if preprocessing_test else '❌ FAIL'}")
    print(f"Full Pipeline:    {'✅ PASS' if pipeline_test else '❌ FAIL'}")

    if all([data_test, preprocessing_test, pipeline_test]):
        print("\n🎉 CONGRATULATIONS! Setup is complete!")
    else:
        print("\n🔧 Some tests failed. Please fix the issues before continuing.")


if __name__ == "__main__":
    main()
