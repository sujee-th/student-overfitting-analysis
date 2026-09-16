"""Create a reproducible synthetic dataset for the overfitting experiment."""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_STUDENTS = 300


def main():
    rng = np.random.default_rng(SEED)
    # Values are designed to resemble plausible weekly study and assessment data.
    study_hours = rng.uniform(1, 20, N_STUDENTS)
    attendance = rng.uniform(45, 100, N_STUDENTS)
    previous_score = rng.normal(65, 15, N_STUDENTS).clip(20, 100)
    assignment_score = rng.normal(68, 18, N_STUDENTS).clip(15, 100)
    online_learning_hours = rng.uniform(0, 15, N_STUDENTS)
    quiz_score = rng.normal(64, 18, N_STUDENTS).clip(10, 100)
    sleep_hours = rng.normal(7, 1.2, N_STUDENTS).clip(4, 10)

    # A hidden academic-performance rule makes labels meaningful. Some random
    # variation reflects real factors that are not included among the features.
    academic_score = (
        0.55 * previous_score + 0.45 * assignment_score + 0.50 * quiz_score
        + 1.25 * study_hours + 0.18 * attendance + 0.45 * online_learning_hours
        - 2.0 * np.abs(sleep_hours - 7) + rng.normal(0, 10, N_STUDENTS)
    )
    threshold = np.median(academic_score)
    performance = (academic_score >= threshold).astype(int)

    # Label noise deliberately represents unmeasured circumstances (health,
    # internet access, etc.). A flexible model can eventually memorize it,
    # which makes this small educational experiment show overfitting.
    flip_indices = rng.choice(N_STUDENTS, size=48, replace=False)
    performance[flip_indices] = 1 - performance[flip_indices]

    data = pd.DataFrame({
        "study_hours": study_hours.round(2),
        "attendance": attendance.round(2),
        "previous_score": previous_score.round(2),
        "assignment_score": assignment_score.round(2),
        "online_learning_hours": online_learning_hours.round(2),
        "quiz_score": quiz_score.round(2),
        "sleep_hours": sleep_hours.round(2),
        "performance": performance,
    })
    project_root = Path(__file__).resolve().parents[1]
    output_path = project_root / "data" / "students.csv"
    output_path.parent.mkdir(exist_ok=True)
    data.to_csv(output_path, index=False)

    print(f"Dataset saved to: {output_path}")
    print(f"Shape: {data.shape[0]} students, {data.shape[1] - 1} features")
    print("Class distribution (0 = At Risk, 1 = Good Performance):")
    print(data["performance"].value_counts().sort_index().to_string())
    print("\nFirst five rows:")
    print(data.head().to_string(index=False))


if __name__ == "__main__":
    main()
