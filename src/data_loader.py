import os
import pandas as pd
from datetime import timedelta

RAW_PATH = os.path.join("data", "raw")
PROCESSED_PATH = os.path.join("data", "processed")

def detect_separator(filepath):
    """Detecta automaticamente o separador do CSV."""
    with open(filepath, "r", encoding="utf-8") as f:
        first_line = f.readline()
    return ";" if ";" in first_line else ","

def load_orders():
    path = os.path.join(RAW_PATH, "Orders_simulated.csv")
    sep = detect_separator(path)
    df = pd.read_csv(path, sep=sep, encoding="utf-8")
    df.columns = df.columns.str.strip().str.lower()

    # Renomear colunas principais
    rename_map = {
        "id": "order_id",
        "customer": "customer_id",
        "createdat": "order_date",
        "saleschannel": "channel",
        "totalamount": "order_amount"
    }
    df = df.rename(columns=rename_map)

    # Conversões de tipo
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", dayfirst=True)
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")

    # Filtrar colunas úteis
    keep_cols = ["order_id", "customer_id", "order_date", "channel", "order_amount"]
    df = df[keep_cols].dropna(subset=["customer_id"])

    return df

def load_customers():
    path = os.path.join(RAW_PATH, "Customer_new_data.csv")
    sep = detect_separator(path)
    df = pd.read_csv(path, sep=sep, encoding="utf-8")
    df.columns = df.columns.str.strip().str.lower()
    if "id" in df.columns:
        df = df.rename(columns={"id": "customer_id"})
    return df

def load_cannoli():
    path = os.path.join(RAW_PATH, "Dados-Cannoli.xlsx")
    df = pd.read_excel(path)
    df.columns = df.columns.str.strip().str.lower()
    if "id" in df.columns:
        df = df.rename(columns={"id": "customer_id"})
    return df

def merge_datasets(orders, customers, cannoli):
    """Combina todas as fontes de dados em um único dataframe de clientes."""
    merged = customers.copy()

    # Métricas de pedidos
    orders_group = (
        orders.groupby("customer_id")
        .agg(
            total_orders=("order_id", "count"),
            total_spent=("order_amount", "sum"),
            avg_order_value=("order_amount", "mean"),
            last_order_date=("order_date", "max")
        )
        .reset_index()
    )

    merged = pd.merge(merged, orders_group, on="customer_id", how="left")

    # Junta informações adicionais do arquivo Cannoli
    if "customer_id" in cannoli.columns:
        merged = pd.merge(merged, cannoli, on="customer_id", how="left")

    # Criação do target supervisionado
    last_date = orders["order_date"].max()
    cutoff_date = last_date - timedelta(days=30)
    active_customers = orders[orders["order_date"] >= cutoff_date]["customer_id"].unique()
    merged["target"] = merged["customer_id"].isin(active_customers).astype(int)

    return merged

def main():
    os.makedirs(PROCESSED_PATH, exist_ok=True)

    print("[INFO] Carregando dados brutos...")
    orders = load_orders()
    customers = load_customers()
    cannoli = load_cannoli()

    print("[INFO] Mesclando datasets...")
    merged = merge_datasets(orders, customers, cannoli)

    print(f"[INFO] Salvando dataset processado com {merged.shape[0]} linhas e {merged.shape[1]} colunas...")
    output_path = os.path.join(PROCESSED_PATH, "clients_training_set.parquet")
    merged.to_parquet(output_path, index=False)
    print(f"[SUCESSO] Dataset salvo em: {output_path}")

if __name__ == "__main__":
    main()
