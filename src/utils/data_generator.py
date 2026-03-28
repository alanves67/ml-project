import numpy as np
import pandas as pd
from typing import Tuple


def generate_synthetic_data(
        n_samples: int = 1000,
        n_features: int = 3,
        noise: float = 0.1,
        random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Генерирует синтетические данные для линейной регрессии"""
    np.random.seed(random_state)

    X = np.random.randn(n_samples, n_features)
    true_coef = np.array([2.0, 3.0, -1.0])
    y = X @ true_coef + noise * np.random.randn(n_samples)

    X_df = pd.DataFrame(
        X,
        columns=[f'feature_{i + 1}' for i in range(n_features)]
    )
    y_series = pd.Series(y, name='target')

    return X_df, y_series