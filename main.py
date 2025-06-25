import sys
import argparse
import subprocess
import os


sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

def run_command(command, description):
    """Runs a command in the terminal and checks for errors."""
    print(f"--- {description} ---")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"✅ SUCCESS: {description} completed.\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ ERROR: Command failed: {' '.join(command)}")
        print(f"Please try running it manually.")
        sys.exit(1) # Exit the script if a setup step fails
    except FileNotFoundError:
        print(f"❌ ERROR: '{command[0]}' komutu bulunamadı. Python veya ilgili programın sistem yolunda (PATH) olduğundan emin olun.")
        sys.exit(1)

def handle_setup():
    """Runs all one-time setup steps for the project."""
    print("🚀 Starting project setup. This only needs to be done once.")
    
    # 1. spaCy modelini indir
    spacy_command = [sys.executable, "-m", "spacy", "download", "en_core_web_md"]
    run_command(spacy_command, "Downloading spaCy model 'en_core_web_md'")
    
    # 2. NLTK verilerini indir (kendi script'imizi çalıştırarak)
    nltk_command = [sys.executable, "-m", "src.download_nltk_data"]
    run_command(nltk_command, "Downloading NLTK data ('stopwords', 'wordnet')")
    
    print("🎉 All setup steps are complete! You are now ready to train the model.")

def handle_train():
    """Imports and runs the main training function."""
    print("🧠 Starting LSTM model training...")
    print("Bu işlem, özellikle ilk çalıştırmada vektörler oluşturulurken uzun sürebilir.")
    try:
        from src.train_lstm import train_sentiment_model
        train_sentiment_model()
    except ImportError as e:
        print(f"❌ ERROR: Could not import training module. Make sure your project structure is correct.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during training: {e}")
        sys.exit(1)

def handle_inspect():
    """Runs a data inspection script."""
    print("🔍 Starting data inspection...")
    print("Bu script, verinin nasıl işlendiğini ve vektörleştiğini gösterir.")
    try:
        from src.tests_and_inspections.test_setup import main as test_data
        test_data()
        from src.tests_and_inspections.inspect_data import main as inspect_data
        inspect_data()
    except ImportError as e:
        print(f"❌ ERROR: Could not import inspection module. Make sure your project structure is correct.")
        print(f"Details: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during inspection: {e}")
        sys.exit(1)


def main():
    # Komut satırı arayüzünü oluştur
    parser = argparse.ArgumentParser(
        description="From-Scratch Sentiment Analysis Project Controller.",
        formatter_class=argparse.RawTextHelpFormatter # Help mesajının formatını korur
    )
    
    parser.add_argument(
        'command', 
        choices=['setup', 'train', 'inspect'], 
        help="The main command to run:\n"
             "  setup    - Downloads all necessary data models from spaCy and NLTK (run this first).\n"
             "  train    - Trains the from-scratch LSTM model.\n"
             "  inspect  - Runs a data inspection script to visualize the data pipeline."
    )
    
    args = parser.parse_args()
    
    # Seçilen komuta göre ilgili fonksiyonu çalıştır
    if args.command == 'setup':
        handle_setup()
    elif args.command == 'train':
        handle_train()
    elif args.command == 'inspect':
        handle_inspect()

if __name__ == "__main__":
    print("======================================================")
    print("==  From-Scratch Sentiment Analysis Control Panel   ==")
    print("======================================================")
    print("Assuming you have installed requirements from requirements.txt and unzipped data into the 'data' folder.\n")
    main()