import os
import argparse
from fnmatch import fnmatch
from huggingface_hub import list_repo_files, hf_hub_download, snapshot_download
from tqdm.contrib.concurrent import thread_map


REPO_ID = "BohdanZasymovych/aircraft-classification"
LOCAL_DIR = "data"


parser = argparse.ArgumentParser(description="Download the data/ directory from Hugging Face Hub")
parser.add_argument("--repo-id", default=REPO_ID, help=f"Hugging Face dataset repo (default: {REPO_ID})")
parser.add_argument("--include", nargs="*", help="Only download matching paths, e.g. 'datasets/russian-planes-6cls/*'")
parser.add_argument("--sync", action="store_true", help="Verify all local files against the Hub and re-download changed ones (slow)")
args = parser.parse_args()

if args.sync:
    snapshot_download(repo_id=args.repo_id,
                      repo_type="dataset",
                      local_dir=LOCAL_DIR,
                      allow_patterns=args.include)
else:
    files = list_repo_files(args.repo_id, repo_type="dataset")
    if args.include:
        files = [f for f in files if any(fnmatch(f, pattern) for pattern in args.include)]
    missing = [f for f in files if not os.path.exists(os.path.join(LOCAL_DIR, f))]

    print(f"{len(files) - len(missing)} files already present, {len(missing)} to download")
    if missing:
        thread_map(lambda f: hf_hub_download(repo_id=args.repo_id, repo_type="dataset", filename=f, local_dir=LOCAL_DIR),
                   missing,
                   max_workers=8,
                   desc="Downloading")

print(f"Data downloaded to {os.path.abspath(LOCAL_DIR)}")
