import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

# Caminhos padrão
PROCESSED_PATH = os.path.join("data", "processed")
MODEL_PATH = os.path.join("models", "classification_model.pkl")


def load_features():
    """Carrega dataset de features."""
    path = os.path.join(PROCESSED_PATH, "clients_features.parquet")
    df = pd.read_parquet(path)
    df.columns = df.columns.str.strip().str.lower()
    print(f"[INFO] Dataset carregado com {df.shape[0]} linhas e {df.shape[1]} colunas.")
    return df


def prepare_data(df):
    """Prepara features numéricas e alvo para treino."""
    if "is_active" not in df.columns:
        raise ValueError("Coluna 'is_active' não encontrada no dataset de features!")

    y = df["is_active"].astype(int)

    drop_cols = [
        "customer_id",
        "loyalty_segment",
        "is_active",
        "is_inactive",
        "top_channel"
    ]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    # Seleciona apenas colunas numéricas
    X = X.select_dtypes(include=[np.number]).fillna(0)

    print(f"[INFO] Dataset de treino preparado com {X.shape[0]} linhas e {X.shape[1]} features.")
    return X, y


def train_model(X, y):
    """Treina modelo de classificação e retorna o modelo e métricas."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    print("[INFO] Treinando modelo...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    metrics = {
        "AUC": roc_auc_score(y_test, y_prob),
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1": f1_score(y_test, y_pred, zero_division=0),
    }

    print("\n[INFO] Resultados do modelo:")
    for k, v in metrics.items():
        print(f"  {k:10s}: {v:.4f}")

    print("\n[INFO] Relatório de Classificação:")
    print(classification_report(y_test, y_pred))

    # Avalia Top 10%
    df_rank = pd.DataFrame({
        "real": y_test,
        "probabilidade": y_prob
    }).sort_values("probabilidade", ascending=False)
    top_10 = df_rank.head(int(len(df_rank) * 0.10))
    uplift_top10 = top_10["real"].mean()

    print(f"\n[INFO] Taxa de recompra no TOP 10% mais propensos: {uplift_top10:.2%}")

    # Salva modelo, scaler e nomes das features
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model": model,
        "scaler": scaler,
        "feature_names": X.columns.tolist()
    }, MODEL_PATH)

    print(f"[SUCESSO] Modelo salvo em: {MODEL_PATH}")
    return model, metrics


def main():
    print("[INFO] Carregando dataset de features...")
    df = load_features()

    X, y = prepare_data(df)
    model, metrics = train_model(X, y)


if __name__ == "__main__":
    main()
