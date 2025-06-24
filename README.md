# 🧠 AI Data Insight

**Projeto completo de Data Science com IA (LLM - Grouk) + Streamlit**

Explore seus dados de forma inteligente com uma interface interativa e recursos de IA para análise, relatório, modelagem e previsões automáticas.

![Streamlit](https://img.shields.io/badge/Streamlit-Online-success?logo=streamlit)
![Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

📤 Executando o App Online
Acesse aqui 👉 [https://ai-data-insight-1.streamlit.app]

## 🚀 Funcionalidades

✅ Upload de arquivos CSV  
✅ Análise automática dos dados  
✅ Estatísticas e gráficos interativos  
✅ Chat com IA (Grouk) sobre o dataset  
✅ Geração de relatório profissional com IA  
✅ Sugestão de código completo de ML com base nos dados  
✅ Execução real de modelos de Regressão ou Classificação  
✅ Download do modelo treinado `.pkl`  
✅ Upload de novos dados e uso do modelo para previsões  
✅ Gráficos de resultados + download em PNG e CSV

---

## 🖥️ Tecnologias Utilizadas

- Python 3.10+
- [Streamlit](https://streamlit.io)
- Pandas
- Scikit-learn
- Matplotlib / Seaborn
- [GROQ API](https://console.groq.com) (compatível com OpenAI SDK)
- Joblib

---

## 📦 Instalação Local

```bash
git clone https://github.com/seu-usuario/ai_data_insight.git
cd ai_data_insight
pip install -r requirements.txt
streamlit run app.py

💡 Crie um arquivo .env ou use Streamlit Cloud Secrets com a variável:
GROQ_API_KEY=sua_chave_da_groq

☁️ Deploy (Streamlit Cloud)
1- Crie um repositório com este projeto

2- Vá em streamlit.io/cloud e conecte com o GitHub

3- Escolha o repositório e defina app.py como entrypoint

4- Em "Advanced settings", adicione os secrets:

GROQ_API_KEY="sua_chave_da_groq"

5- Clique em Deploy ✅


🤝 Contribuições
Sinta-se à vontade para abrir issues, forks ou PRs.
Esse projeto é ideal para quem quer entender como aplicar IA no cotidiano de um cientista de dados.

📄 Licença
MIT License

