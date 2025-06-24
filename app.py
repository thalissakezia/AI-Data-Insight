import streamlit as st
import pandas as pd
from data_utils import analisar_dados, rodar_modelo
from grouk_helper import perguntar_para_ia, gerar_relatorio_ia, sugerir_modelo_e_codigo
import joblib
import io

st.set_page_config(page_title="AI Data Insight", layout="wide")
st.title("📊 AI Data Insight")
st.caption("Explore seus dados com ajuda de IA (Grouk API)")

# Upload do arquivo CSV
uploaded_file = st.file_uploader("📁 Faça upload de um arquivo CSV", type=["csv"])

df = None

# Se um arquivo for enviado
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)

        st.success("✅ Arquivo carregado com sucesso!")
        st.subheader("🧮 Visualização dos Dados")
        st.dataframe(df.head(100), use_container_width=True)

        st.subheader("📈 Estatísticas Básicas")
        stats = analisar_dados(df)
        st.write(stats)

        st.info("Na próxima etapa, você poderá conversar com a IA sobre seus dados.")

        st.subheader("💬 Chat com a IA sobre os dados")

        pergunta_usuario = st.text_area("Digite uma pergunta sobre o dataset:")

        if st.button("Perguntar à IA") and pergunta_usuario:
            with st.spinner("Consultando a IA..."):
                contexto = df.head(50).to_string()  # você pode ajustar o tamanho do contexto
                resposta = perguntar_para_ia(pergunta_usuario, contexto)
                st.success("Resposta da IA:")
                st.write(resposta)

    except Exception as e:
        st.error(f"❌ Erro ao carregar o arquivo: {e}")
else:
    st.warning("🔺 Faça upload de um arquivo CSV para começar.")
st.subheader("📄 Gerar Relatório Automático")

if st.button("Gerar relatório com IA") and df is not None:
    with st.spinner("A IA está gerando o relatório..."):
        relatorio = gerar_relatorio_ia(df)
        st.text_area("📝 Relatório Gerado", value=relatorio, height=300)

        # Download do relatório
        st.download_button("📥 Baixar relatório (.txt)", relatorio, file_name="relatorio_ia.txt")


if st.button("Gerar sugestão de modelo + código") and df is not None:
    with st.spinner("A IA está analisando os dados e gerando código..."):
        codigo_gerado = sugerir_modelo_e_codigo(df)
        st.code(codigo_gerado, language='python')

        st.download_button("📥 Baixar código sugerido", codigo_gerado, file_name="modelo_ia.py")

st.subheader("📤 Fazer Previsões com Modelo Treinado")

uploaded_model = st.file_uploader("🔁 Faça upload de um modelo `.pkl`", type=["pkl"])
uploaded_prediction_data = st.file_uploader("📁 Faça upload de novos dados para previsão (CSV)", type=["csv"])

if uploaded_model and uploaded_prediction_data:
    try:
        # Carrega o modelo
        modelo_carregado = joblib.load(uploaded_model)

        # Lê os dados
        df_novo = pd.read_csv(uploaded_prediction_data)

        # Ajusta as colunas para bater com as do modelo
        try:
            features_treinadas = modelo_carregado.feature_names_in_
            df_numerico = df_novo[features_treinadas]
        except AttributeError:
            # Se o modelo não tiver feature_names_in_, mantém só as numéricas
            df_numerico = df_novo.select_dtypes(include=["number"])

        st.info("✅ Modelo e dados carregados com sucesso!")

        # Faz as previsões
        predicoes = modelo_carregado.predict(df_numerico)
        df_novo["Previsão"] = predicoes

        st.subheader("🔮 Resultados das Previsões")
        st.dataframe(df_novo, use_container_width=True)

        # Baixar resultados
        csv_buffer = io.StringIO()
        df_novo.to_csv(csv_buffer, index=False)
        st.download_button("📥 Baixar resultados com previsões (.csv)", csv_buffer.getvalue(), file_name="resultados_com_previsoes.csv")

    except Exception as e:
        st.error(f"Erro ao aplicar o modelo: {e}")

if df is not None:
    st.subheader("⚙️ Executar Modelo no App")
    target_options = df.columns.tolist()
    target = st.selectbox("Selecione a variável alvo (target):", target_options)
    if st.button("Executar modelo com esse target"):
        with st.spinner("Treinando e avaliando o modelo..."):
            rodar_modelo(df, target)
