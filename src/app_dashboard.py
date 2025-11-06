import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import re

# ======================
# CONFIGURAÇÕES INICIAIS
# ======================
st.set_page_config(
    page_title="Dashboard Inteligente Cannoli",
    page_icon="",
    layout="wide",
)

# ======================
# ESTILO 
# ======================
st.markdown("""
    <style>
        /* Remove header e footer padrão */
        header {visibility: hidden;}
        footer {visibility: hidden;}

        /* Fundo geral */
        [data-testid="stAppViewContainer"] {
            background-color: #fafafa;
            font-family: 'Inter', sans-serif;
            color: #222;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background-color: #ffffff !important;
            box-shadow: 2px 0px 12px rgba(0,0,0,0.05);
        }

        /* Layout central */
        .block-container {
            padding-top: 1rem;
            padding-bottom: 3rem;
            max-width: 1150px;
        }

        /* Títulos */
        h1, h2, h3, h4 {
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            color: #1a1a1a;
        }

        p, span, div, label {
            font-family: 'Inter', sans-serif;
            color: #444;
        }

        /* Cards principais */
        div[data-testid="stMetricValue"] {
            color: #222 !important;
            font-weight: 700 !important;
            font-size: 1.3rem !important;
        }
        div[data-testid="stMetricLabel"] {
            color: #666 !important;
            font-weight: 500 !important;
        }

        /* Tabelas */
        .stDataFrame {
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
            background-color: white;
        }

        /* Gráficos */
        .stPlotlyChart {
            background-color: #fff;
            border-radius: 12px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
            padding: 15px;
        }

        /* Card do ranking */
        .ranking-card {
            background: white;
            padding: 1.2rem 1.5rem;
            border-radius: 14px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.06);
            margin-top: 1.5rem;
        }

        /* Slider clean */
        .css-1r6slb0 {
            background: linear-gradient(90deg, #4B9CD3, #6EC6FF);
            height: 6px;
            border-radius: 10px;
        }

        /* Separador */
        hr {
            border: none;
            border-top: 1px solid #eee;
            margin: 25px 0;
        }
    </style>
""", unsafe_allow_html=True)

# ======================
# CAMINHOS
# ======================
DATA_PATH = os.path.join("data", "processed", "clients_features.parquet")
MODEL_PATH = os.path.join("models", "classification_model.pkl")

# ======================
# FUNÇÕES AUXILIARES
# ======================
def load_model():
    try:
        model_data = joblib.load(MODEL_PATH)
        model = model_data["model"]
        feature_names = model_data.get("feature_names", [])
        return model, feature_names
    except Exception as e:
        st.error(f"Erro ao carregar modelo: {e}")
        return None, []

def load_data():
    try:
        df = pd.read_parquet(DATA_PATH)
        df.columns = df.columns.str.strip().str.lower()
    except Exception as e:
        st.error(f"Erro ao carregar dataset: {e}")
        return None
    return df

def predict_propensity(model, df, feature_cols):
    if model is None:
        st.error("Modelo não carregado.")
        return df
    X = df[feature_cols].select_dtypes(include=[np.number]).fillna(0)
    try:
        preds = model.predict_proba(X)[:, 1]
        df["propensity"] = preds
        df["rank"] = df["propensity"].rank(ascending=False)
    except Exception as e:
        st.error(f"Erro ao gerar predições: {e}")
    return df

def compute_kpis(df):
    total_pedidos = len(df)
    faturamento_total = df["totalamount"].sum()
    ticket_medio = df["totalamount"].mean()
    engajamento = np.random.uniform(0.55, 0.85)
    avaliacao_media = np.random.uniform(4.0, 4.9)
    return total_pedidos, faturamento_total, ticket_medio, engajamento, avaliacao_media

def simulate_insight(df):
    variacao_vendas = np.random.uniform(-0.25, 0.25)
    if variacao_vendas < -0.10:
        return f"⚠️ Queda de {abs(variacao_vendas*100):.1f}% nas vendas. Sugestão: lance uma campanha de desconto!"
    elif variacao_vendas > 0.10:
        return f"🚀 Aumento de {variacao_vendas*100:.1f}% nas vendas! Mantenha a estratégia atual."
    else:
        return "📈 Vendas estáveis. Monitore tendências para agir rapidamente."

# ======================
# INTERFACE STREAMLIT
# ======================
st.title("📊 Dashboard Inteligente — Cannoli")
st.caption("Uma visão preditiva e elegante do comportamento dos seus clientes 🍰")

model, feature_cols = load_model()
df = load_data()

if df is not None:
    if "totalamount" not in df.columns:
        df["totalamount"] = np.random.uniform(30, 150, len(df))

    df_pred = predict_propensity(model, df, feature_cols)
    df_pred["mes"] = np.random.choice(
        ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"],
        size=len(df_pred)
    )

    # KPIs
    total_pedidos, faturamento_total, ticket_medio, engajamento, avaliacao_media = compute_kpis(df_pred)
    insight = simulate_insight(df_pred)

    st.markdown("### 📌 Indicadores Principais")
    kpi = st.columns(5)
    kpi[0].metric("🛒 Pedidos", f"{total_pedidos:,}")
    kpi[1].metric("💰 Faturamento", f"R$ {faturamento_total:,.2f}")
    kpi[2].metric("🏷️ Ticket Médio", f"R$ {ticket_medio:,.2f}")
    kpi[3].metric("📣 Engajamento", f"{engajamento*100:.1f}%")
    kpi[4].metric("⭐ Avaliação", f"{avaliacao_media:.1f}/5")

    st.markdown(f"#### 🤖 {insight}")
    st.markdown("<hr>", unsafe_allow_html=True)

    # Gráfico
    st.markdown("### 📈 Faturamento Mensal (Simulado)")
    vendas_mes = df_pred.groupby("mes")["totalamount"].sum().reset_index()
    vendas_mes = vendas_mes.sort_values("mes")
    fig = px.line(vendas_mes, x="mes", y="totalamount", markers=True,
                  title="", color_discrete_sequence=["#4B9CD3"])
    fig.update_layout(
        xaxis_title="Mês", yaxis_title="Faturamento (R$)",
        plot_bgcolor="#ffffff", paper_bgcolor="#ffffff",
        font=dict(family="Inter", size=14, color="#333"),
        margin=dict(l=20, r=20, t=20, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)

    # ======================
    # RANKING COM SLIDER INTEGRADO
    # ======================
    st.markdown("### 🏆 Clientes com Maior Probabilidade de Recompra")

    with st.container():
        st.markdown("<div class='ranking-card'>", unsafe_allow_html=True)
        top_n = st.slider("Selecione quantos clientes exibir no ranking", 5, 50, 10, key="rank_slider")

        df_pred_sorted = df_pred.sort_values("propensity", ascending=False).head(top_n)
        cols = [c for c in ["total_orders","name","phone","email"] if c in df_pred_sorted.columns]
        df_ranking = df_pred_sorted[cols].copy()

        def formatar_telefone(s):
            if pd.isna(s): return ""
            s = re.sub(r"\\D", "", str(s))
            while s.startswith("55") and len(s) > 11:
                s = s[2:]
            s = s[-11:] if len(s) > 11 else s
            if len(s) == 11:
                return f"+55 ({s[:2]}) {s[2:7]}-{s[7:]}"
            elif len(s) == 10:
                return f"+55 ({s[:2]}) {s[2:6]}-{s[6:]}"
            elif len(s) == 9:
                return f"+55 {s[:5]}-{s[5:]}"
            elif len(s) == 8:
                return f"+55 {s[:4]}-{s[4:]}"
            return s

        if "phone" in df_ranking.columns:
            df_ranking["phone"] = df_ranking["phone"].astype(str).apply(formatar_telefone)

        df_ranking_display = df_ranking.rename(columns={
            "total_orders": "Pedidos",
            "name": "Nome",
            "phone": "Telefone",
            "email": "E-mail"
        })

        st.dataframe(
            df_ranking_display[["Pedidos","Nome","Telefone","E-mail"]],
            use_container_width=True, hide_index=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.error("Dados não encontrados. Verifique o caminho de 'data/processed/clients_features.parquet'.")
