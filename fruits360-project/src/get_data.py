# src/get_data.py
import kagglehub, os
from pathlib import Path

# Download the dataset (cached under ~/.cache/kagglehub)
path = Path(kagglehub.dataset_download("moltean/fruits")).resolve()

# Common branches you might want:
b100 = path / "fruits-360_100x100"
orig = path / "fruits-360_original-size"
three = path / "fruits-360_3-body-problem"

print("KaggleHub root:", path)
print("100x100:", b100 if b100.exists() else "NOT FOUND")
print("original-size:", orig if orig.exists() else "NOT FOUND")
print("3-body-problem:", three if three.exists() else "NOT FOUND")

# Optional: create a convenient symlink under ./data so the rest of your code can refer to it
project_data = Path(__file__).resolve().parents[1] / "data"
project_data.mkdir(exist_ok=True)

# Change which branch you prefer to link by editing BRANCH below:
BRANCH = b100  # or orig / three
link = project_data / BRANCH.name
if link.exists() or link.is_symlink():
    link.unlink()
link.symlink_to(BRANCH)
print(f"Linked {BRANCH} -> {link}")
