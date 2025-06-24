import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
import io

def analisar_dados(df: pd.DataFrame) -> dict:
    resumo = {}
    resumo["Dimensões"] = f"{df.shape[0]} linhas x {df.shape[1]} colunas"
    resumo["Colunas"] = list(df.columns)
    resumo["Tipos de Dados"] = df.dtypes.apply(lambda x: str(x)).to_dict()
    resumo["Valores Ausentes (%)"] = df.isnull().mean().round(2).multiply(100).to_dict()
    try:
        resumo["Descrição Estatística"] = df.describe(include='all', datetime_is_numeric=True).to_dict()
    except TypeError:
        resumo["Descrição Estatística"] = df.describe(include='all').to_dict()
    return resumo

def rodar_modelo(df, target_col):
    # SELEÇÃO DE COLUNAS PREDITIVAS
    colunas_usaveis = st.multiselect(
        "Selecione as colunas preditoras:",
        df.drop(columns=[target_col]).columns.tolist(),
        default=df.drop(columns=[target_col]).select_dtypes(include=["number"]).columns.tolist()
    )
    if not colunas_usaveis:
        st.warning("Selecione pelo menos uma coluna para treinar o modelo.")
        return

    X = df[colunas_usaveis]
    y = df[target_col]

    # Remove colunas não numéricas (para simplificar)
    X = X.select_dtypes(include=["number"])

    # Verifica se é regressão ou classificação
    if y.nunique() <= 10 and y.dtype in ["int64", "int32", "object"]:
        problema = "classificacao"
        modelo = LogisticRegression(max_iter=1000)
    else:
        problema = "regressao"
        modelo = LinearRegression()

    try:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        st.success(f"✅ Modelo de {problema} treinado com sucesso!")

        if problema == "regressao":
            mse = mean_squared_error(y_test, y_pred)
            st.metric("Erro Quadrático Médio (MSE)", f"{mse:.2f}")
            fig, ax = plt.subplots()
            sns.scatterplot(x=y_test, y=y_pred, ax=ax)
            ax.set_xlabel("Valor Real")
            ax.set_ylabel("Predito")
            ax.set_title("Regressão: Valor Real vs Predito")
            st.pyplot(fig)

            # Salvar gráfico como PNG para download
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            st.download_button("🖼️ Baixar gráfico", img_buffer, file_name="grafico_resultado.png")

        else:
            y_pred_labels = modelo.predict(X_test)
            acc = accuracy_score(y_test, y_pred_labels)
            st.metric("Acurácia", f"{acc * 100:.2f}%")
            fig, ax = plt.subplots()
            sns.countplot(x=y_pred_labels, ax=ax)
            ax.set_title("Distribuição das Previsões")
            st.pyplot(fig)

            # Salvar gráfico como PNG para download
            img_buffer = io.BytesIO()
            fig.savefig(img_buffer, format='png')
            img_buffer.seek(0)
            st.download_button("🖼️ Baixar gráfico", img_buffer, file_name="grafico_resultado.png")

        # Exportar modelo treinado em memória
        buffer = io.BytesIO()
        joblib.dump(modelo, buffer)
        buffer.seek(0)
        st.download_button("📦 Baixar modelo treinado (.pkl)", buffer, file_name="modelo_treinado.pkl")

    except Exception as e:
        st.error(f"Erro ao treinar modelo: {e}")
