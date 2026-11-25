"""Data preparation step (Review 2): inspect the Hugging Face dataset.

Prints rows, columns, number of classes and class counts, checks which dataset
labels have a verified meaning in our reference list, and saves:
  results/dataset_classes.csv   - every class, its count and reference match
  results/dataset_samples.png   - a grid of example images

Run:  python explore_dataset.py
"""
import pandas as pd
from PIL import Image, ImageDraw

from src import config, dataset, reference


def main():
    info = dataset.describe()
    print(f"Dataset: {config.HF_DATASET}")
    print(f"Rows: {info['rows']} | Classes: {info['num_classes']} | "
          f"Image column: {info['image_column']} | Label column: {info['label_column']}")

    rows = []
    for name, count in info["class_counts"].items():
        r = reference.lookup(name)
        rows.append({"class": name, "images": count, "reference_match": r.name if r.found else "",
                     "match_score": r.score})
    df = pd.DataFrame(rows)
    config.RESULTS_DIR.mkdir(exist_ok=True)
    df.to_csv(config.RESULTS_DIR / "dataset_classes.csv", index=False)

    covered = (df.reference_match != "").sum()
    print(f"\nClasses with a verified reference meaning: {covered}/{len(df)}")
    print(f"Largest class: {df.iloc[0]['class']} ({df.iloc[0]['images']}) | "
          f"Smallest: {df.iloc[-1]['class']} ({df.iloc[-1]['images']})")
    print("\nClasses WITHOUT a reference meaning (add them to data/reference_signs.json):")
    for c in df[df.reference_match == ""]["class"]:
        print("  -", c)

    # sample grid
    samples = dataset.sample(16, seed=1)
    cell = 160
    grid = Image.new("RGB", (cell * 4, (cell + 24) * 4), "white")
    d = ImageDraw.Draw(grid)
    for i, (img, lab) in enumerate(samples):
        x, y = (i % 4) * cell, (i // 4) * (cell + 24)
        grid.paste(img.resize((cell - 8, cell - 8)), (x + 4, y + 4))
        d.text((x + 4, y + cell), lab[:26], fill="black")
    grid.save(config.RESULTS_DIR / "dataset_samples.png")
    print("\nSaved results/dataset_classes.csv and results/dataset_samples.png")


if __name__ == "__main__":
    main()
