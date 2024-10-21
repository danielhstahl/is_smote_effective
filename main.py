import numpy as np
import random
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import MiniBatchKMeans
from typing import List
from imblearn.over_sampling import KMeansSMOTE

LOGIT = [7.0, 5.0, 3.0, -0.5]


def actual_logit(x1: float, x2: float, x3: float) -> float:
    return LOGIT[0] + LOGIT[1] * x1 + LOGIT[2] * x2 + LOGIT[3] * x3


def convert_logit_to_probability(logit: float) -> float:
    elogit = np.exp(logit)
    return elogit / (1 + elogit)


def simulate_coin_flip(p: float) -> int:
    x = random.random()
    if x > p:
        return 1
    return 0


def get_x(row_size: int, feature_size: int) -> np.array:
    return np.random.normal(loc=0.0, scale=0.5, size=(row_size, feature_size))


def simulate_y(x: np.array) -> np.array:
    simulate_logistic = lambda row: simulate_coin_flip(
        convert_logit_to_probability(actual_logit(row[0], row[1], row[2]))
    )
    return np.apply_along_axis(simulate_logistic, axis=1, arr=x)


def estimate(X: np.array, y: np.array):
    clf = LogisticRegression(random_state=0).fit(X, y)
    return clf


def get_squared_error_in_coefs(intercept: float, coefficients: List[float]) -> float:
    sq_error = 0
    for actual, estimate in zip(LOGIT, [intercept] + coefficients):
        diff = actual - estimate
        sq_error += diff * diff
    return sq_error


def simulation_standard(data_size: int) -> float:
    data_size = 10000
    X = get_x(data_size, len(LOGIT) - 1)

    y = simulate_y(X)
    fitted_lr = estimate(X, y)
    return get_squared_error_in_coefs(fitted_lr.intercept_, fitted_lr.coef_[0].tolist())


def simulation_smote(data_size: int) -> float:
    data_size = 10000
    X = get_x(data_size, len(LOGIT) - 1)
    y = simulate_y(X)

    X_resampled, y_resampled = KMeansSMOTE(
        cluster_balance_threshold=0.3,
        kmeans_estimator=MiniBatchKMeans(n_clusters=100, n_init=3),
    ).fit_resample(X, y)
    fitted_lr = estimate(X_resampled, y_resampled)
    return get_squared_error_in_coefs(fitted_lr.intercept_, fitted_lr.coef_[0].tolist())


if __name__ == "__main__":
    data_size = 10000
    num_sim = 100

    average_error_smote = 0
    average_error_standard = 0
    for i in range(num_sim):
        average_error_smote += simulation_smote(data_size)
        average_error_standard += simulation_standard(data_size)
    # This simulation suggests that SMOTE is not effective at
    # improving the estimate of the underlying coefficients
    print("Average Error Smote:", average_error_smote / num_sim)
    print("Average Error Sttandard:", average_error_standard / num_sim)
