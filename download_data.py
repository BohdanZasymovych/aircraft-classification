import argparse
from huggingface_hub import snapshot_download


REPO_ID = "BohdanZasymovych/aircraft-classification"
LOCAL_DIR = "data"


parser = argparse.ArgumentParser(description="Download the data/ directory from Hugging Face Hub")
parser.add_argument("--repo-id", default=REPO_ID, help=f"Hugging Face dataset repo (default: {REPO_ID})")
parser.add_argument("--include", nargs="*", help="Only download matching paths, e.g. 'datasets/russian-planes-6cls/*'")
args = parser.parse_args()

path = snapshot_download(repo_id=args.repo_id,
                         repo_type="dataset",
                         local_dir=LOCAL_DIR,
                         allow_patterns=args.include)

print(f"Data downloaded to {path}")
