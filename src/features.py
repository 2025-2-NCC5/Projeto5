import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

PROCESSED_PATH = os.path.join("data", "processed")

def load_dataset():
    path = os.path.join(PROCESSED_PATH, "clients_training_set.parquet")
    df = pd.read_parquet(path)
    df.columns = df.columns.str.strip().str.lower()
    return df

def compute_rfm(df):
    """Calcula métricas RFM: Recência, Frequência e Valor Monetário."""
    reference_date = df["last_order_date"].max()
    df["recency_days"] = (reference_date - df["last_order_date"]).dt.days.fillna(9999)
    df["frequency"] = df["total_orders"].fillna(0)
    df["monetary_value"] = df["avg_order_value"].fillna(0)

    # Normaliza escalas (para ajudar no modelo)
    df["recency_score"] = pd.qcut(df["recency_days"], 5, labels=False, duplicates="drop")
    df["frequency_score"] = pd.qcut(df["frequency"].rank(method="first"), 5, labels=False, duplicates="drop")
    df["monetary_score"] = pd.qcut(df["monetary_value"].rank(method="first"), 5, labels=False, duplicates="drop")

    # Score RFM combinado
    df["rfm_score"] = (
        df["recency_score"].astype(int)
        + df["frequency_score"].astype(int)
        + df["monetary_score"].astype(int)
    )

    return df

def compute_engagement_features(df):
    """Gera atributos de engajamento e canal dominante."""
    if "channel" in df.columns:
        # Contagem por canal
        channel_counts = df.groupby("channel")["customer_id"].count().reset_index()
        channel_counts = channel_counts.rename(columns={"customer_id": "channel_orders"})
        top_channel = channel_counts.sort_values("channel_orders", ascending=False).head(1)["channel"].values[0]
        df["top_channel"] = top_channel
    else:
        df["top_channel"] = "unknown"

    # Cliente ativo ou não
    df["is_active"] = np.where(df["recency_days"] <= 30, 1, 0)
    df["is_inactive"] = np.where(df["recency_days"] > 90, 1, 0)

    # Engajamento (quanto mais pedidos e mais recentes, maior)
    df["engagement_score"] = (
        (5 - df["recency_score"].astype(float))
        + df["frequency_score"].astype(float)
        + df["monetary_score"].astype(float)
    )

    return df

def compute_trends(df):
    """Cria variáveis de tendência (ex: variação de ticket médio, queda de pedidos)."""
    if "total_orders" in df.columns and "avg_order_value" in df.columns:
        df["order_growth_est"] = np.log1p(df["total_orders"]) / df["total_orders"].max()
        df["spending_index"] = df["avg_order_value"] / df["avg_order_value"].mean()
    else:
        df["order_growth_est"] = np.nan
        df["spending_index"] = np.nan

    df["loyalty_segment"] = pd.cut(
        df["rfm_score"],
        bins=[-1, 4, 7, 10, 12],
        labels=["churned", "neutral", "loyal", "vip"]
    )
    return df

def summarize_features(df):
    print("\n[INFO] Features geradas:")
    print(f"- Total de clientes: {df.shape[0]}")
    print(f"- Colunas finais: {len(df.columns)}")
    print(f"- Segmentos de fidelidade: {df['loyalty_segment'].value_counts(dropna=False).to_dict()}")

def main():
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    print("[INFO] Carregando dataset processado...")
    df = load_dataset()

    print("[INFO] Gerando métricas RFM...")
    df = compute_rfm(df)

    print("[INFO] Criando features de engajamento...")
    df = compute_engagement_features(df)

    print("[INFO] Calculando tendências e segmentos...")
    df = compute_trends(df)

    summarize_features(df)

    output_path = os.path.join(PROCESSED_PATH, "clients_features.parquet")
    df.to_parquet(output_path, index=False)
    print(f"[SUCESSO] Arquivo salvo em: {output_path}")

if __name__ == "__main__":
    main()
