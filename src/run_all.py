"""Run the entire overfitting assignment from one VS Code Run button."""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parents[1]
    scripts = [
        "src/generate_data.py",
        "src/train_model.py",
        "src/evaluate_model.py",
    ]
    print("Starting the complete Student Overfitting Analysis...\n")
    for script in scripts:
        print(f"{'=' * 60}\nRunning: {script}\n{'=' * 60}")
        subprocess.run([sys.executable, script], cwd=root, check=True)
    print("\nComplete. Open outputs/ to view graphs and results.txt.")


if __name__ == "__main__":
    main()
