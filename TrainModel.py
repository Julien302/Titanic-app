"""TrainModel.py
Script d'entraînement pour le projet Titanic.
- Charge les données `train.csv`
- Prépare les variables (encodage, imputation)
- Entraîne un RandomForestClassifier
- Affiche l'accuracy sur le jeu de test
- Sauvegarde le modèle dans `model.joblib`
"""

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import joblib

DATA_PATH = Path(__file__).resolve().parent / "train.csv"
MODEL_PATH = Path(__file__).resolve().parent / "model.joblib"
RANDOM_STATE = 123

def load_data(path: Path) -> pd.DataFrame:
    """Charge le fichier CSV Titanic."""
    return pd.read_csv(path)

def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """Renvoie un transformeur qui prépare X."""
    numeric_cols = ["Age", "Fare", "SibSp", "Parch"]
    categorical_cols = ["Pclass", "Sex", "Embarked"]

    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipeline, numeric_cols),
        ("cat", cat_pipeline, categorical_cols),
    ])

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    clf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    return clf

def main():
    print("\n=== Entraînement du modèle Titanic ===")

    df = load_data(DATA_PATH)
    y = df["Survived"]
    X = df.drop(columns=["Survived", "PassengerId", "Name", "Ticket", "Cabin"])

    preprocessor = build_preprocessor(df)
    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = train_random_forest(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Accuracy sur le jeu de test : {acc:.3f}")

    # Sauvegarde à la fois le modèle et le préprocesseur dans un dict
    artifact = {
        "model": model,
        "preprocessor": preprocessor,
    }
    joblib.dump(artifact, MODEL_PATH)
    print(f"Modèle et préprocesseur sauvegardés -> {MODEL_PATH.relative_to(Path.cwd())}")

if __name__ == "__main__":
    main()
