import streamlit as st
from groq import Groq
import pandas as pd

# Pegando a chave direto dos secrets do Streamlit
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

def perguntar_para_ia(pergunta, contexto):
    prompt = (
        "Você é um assistente de dados. Responda à pergunta do usuário com base no contexto do dataset.\n\n"
        f"Contexto:\n{contexto}\n\nPergunta: {pergunta}"
    )
    messages = [
        {"role": "system", "content": "Você é um cientista de dados experiente."},
        {"role": "user", "content": prompt}
    ]
    try:
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao consultar IA: {e}"

def gerar_relatorio_ia(df: pd.DataFrame) -> str:
    try:
        contexto = df.head(50).to_string()
        prompt = (
            "Gere um relatório profissional em linguagem natural com base neste dataset. "
            "Identifique padrões, colunas importantes, valores ausentes e sugira possíveis análises ou modelos de machine learning.\n\n"
            f"Dados:\n{contexto}"
        )
        messages = [
            {"role": "system", "content": "Você é um cientista de dados experiente que escreve relatórios claros."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao gerar relatório: {e}"

def sugerir_modelo_e_codigo(df: pd.DataFrame) -> str:
    try:
        contexto = df.head(50).to_string()
        prompt = (
            "Com base neste dataset, identifique automaticamente qual variável pode ser o alvo "
            "(target) para um modelo supervisionado e gere um código Python completo e funcional "
            "usando Scikit-learn para treinar e avaliar um modelo adequado (classificação ou regressão). "
            "Inclua separação de treino/teste, tratamento de dados se necessário e avaliação simples.\n\n"
            f"Dados:\n{contexto}"
        )
        messages = [
            {"role": "system", "content": "Você é um cientista de dados especialista em machine learning."},
            {"role": "user", "content": prompt}
        ]
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=messages,
            temperature=0.4
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Erro ao gerar sugestão de modelo: {e}"
