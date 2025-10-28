import kagglehub
import pathlib
import shutil
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
TARGET = DATA_DIR / "fruits-360_100x100"  # We normalize to this path

def main():
    print("Downloading Fruits-360 via kagglehub…")
    base_path = pathlib.Path(kagglehub.dataset_download("moltean/fruits"))
    print(f"KaggleHub base: {base_path}")

    # Find the 100x100 branch inside the downloaded cache
    candidates = list(base_path.rglob("fruits-360_100x100"))
    if not candidates:
        # Fallback: look for a Training folder under any path that has '100x100'
        candidates = [p.parent for p in base_path.rglob("Training") if "100x100" in str(p)]
    if not candidates:
        print("ERROR: Could not locate 'fruits-360_100x100' in the downloaded dataset.")
        sys.exit(1)

    src_100 = candidates[0]
    print(f"Found 100x100 branch at: {src_100}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # If existing, remove to refresh the pointer/copy
    if TARGET.exists() or TARGET.is_symlink():
        try:
            if TARGET.is_symlink():
                TARGET.unlink()
            else:
                shutil.rmtree(TARGET)
        except Exception as e:
            print(f"Warning: could not clean existing target: {e}")

    # Prefer symlink to avoid duplicating data
    try:
        TARGET.symlink_to(src_100, target_is_directory=True)
        print(f"Created symlink: {TARGET} -> {src_100}")
    except (OSError, NotImplementedError) as e:
        print(f"Symlink not available ({e}); copying files. This may take a while…")
        shutil.copytree(src_100, TARGET)
        print(f"Copied dataset to: {TARGET}")

    # Basic sanity
    train_dir = TARGET / "Training"
    test_dir = TARGET / "Test"
    if not train_dir.exists() or not test_dir.exists():
        print("ERROR: Expected 'Training/' and 'Test/' under fruits-360_100x100.")
        sys.exit(1)

    print("Dataset ready at:", TARGET)

if __name__ == "__main__":
    main()
