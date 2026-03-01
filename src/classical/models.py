from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def build_estimator(name: str, seed: int = 42):
    key = name.lower()
    if key == "svm":
        return SVC(kernel="rbf", C=5.0, gamma="scale", probability=True, random_state=seed)
    if key == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    if key in {"nb", "naive_bayes"}:
        return GaussianNB()
    raise ValueError("Unsupported estimator. Use one of: svm, rf, nb")
