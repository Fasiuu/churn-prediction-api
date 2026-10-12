from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", autouse=True)
def trained_model():
    """Train the model once if it isn't there yet (e.g. a fresh clone or CI run)."""
    if not (ROOT / "models" / "churn_pipeline.joblib").exists():
        import train
        train.main()
