# src/get_data.py
import kagglehub
import pathlib
import shutil
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TARGET = DATA_DIR / "fruits-360_100x100"  # normalized local path

def find_leaf_with_train_test(root: pathlib.Path):
    """Return the first directory under root that has both Training/ and Test/."""
    for p in root.rglob("*"):
        if p.is_dir():
            train = p / "Training"
            test = p / "Test"
            if train.exists() and test.exists():
                return p
    return None

def main():
    print("Downloading Fruits-360 via kagglehub…")
    base = pathlib.Path(kagglehub.dataset_download("moltean/fruits"))
    print("KaggleHub base:", base)

    # Prefer an obvious 100x100 container, but we'll verify it has Training/Test
    candidates = list(base.rglob("fruits-360_100x100"))
    if not candidates:
        # Fallback: any dir whose path mentions 100x100
        candidates = [p for p in base.rglob("*") if p.is_dir() and "100x100" in str(p)]
    if not candidates:
        print("ERROR: Could not find a '100x100' branch in the downloaded dataset.")
        sys.exit(1)

    # From the candidate, find the actual leaf dir that contains Training/Test
    leaf = None
    for c in candidates:
        leaf = find_leaf_with_train_test(c)
        if leaf:
            break
    if not leaf:
        # Try from the dataset base as a final fallback
        leaf = find_leaf_with_train_test(base)
    if not leaf:
        print("ERROR: Expected a folder that contains 'Training/' and 'Test/' but could not find one.")
        sys.exit(1)

    print("Resolved 100x100 leaf:", leaf)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Clean existing target
    if TARGET.exists() or TARGET.is_symlink():
        try:
            if TARGET.is_symlink():
                TARGET.unlink()
            else:
                shutil.rmtree(TARGET)
        except Exception as e:
            print(f"Warning: could not clean existing target: {e}")

    # Prefer symlink (fast). Fall back to copy if symlinks not allowed.
    try:
        TARGET.symlink_to(leaf, target_is_directory=True)
        print(f"Created symlink: {TARGET} -> {leaf}")
    except (OSError, NotImplementedError) as e:
        print(f"Symlink not available ({e}); copying files. This may take a while…")
        shutil.copytree(leaf, TARGET)
        print(f"Copied dataset to: {TARGET}")

    # Final sanity
    if not (TARGET / "Training").exists() or not (TARGET / "Test").exists():
        print("ERROR: After linking/copying, 'Training/' and 'Test/' are still missing.")
        sys.exit(1)

    print("Dataset ready at:", TARGET)

if __name__ == "__main__":
    main()
