# 🎵 Projeto NLP – Análise Automatizada de Comentários do YouTube  
UFABC — Processamento de Linguagem Natural  

Este projeto realiza **coleta, análise e geração de relatórios** sobre comentários de vídeos do YouTube, empregando técnicas modernas de PLN (Processamento de Linguagem Natural) e modelos LLM (Groq e OpenAI).  

O objetivo principal é entender padrões emocionais, sentimentais e contextuais em comentários musicais.

---

## 🚀 Funcionalidades Principais

### **1. Coleta automática de comentários do YouTube**
- Usa a API oficial do YouTube Data v3.  
- Recupera até *N* comentários por vídeo.  
- Permite escolher a ordem de coleta:
  - `relevance`
  - `time`
  - `rating`

---

### **2. Processamento avançado dos comentários**
Cada comentário passa por:

- Conversão de emojis → texto
- Detecção de idioma (ISO-639-1)
- Tradução automática para PT-BR (se necessário)
- Classificação:
  - **Sentimento** (positivo, negativo, neutro)
  - **Emoção dominante** (ex.: alegria, tristeza, nostalgia…)
  - **Tipo de contexto**:
    - sobre_a_musica  
    - experiencia_pessoal  
    - trecho_de_letra  
    - off_topic
- Extração de **5–10 palavras-chave relevantes**

---

### **3. Estatísticas consolidadas**

O sistema gera distribuições de:

- Sentimentos  
- Emoções  
- Tipos de contexto  
- Idiomas detectados  
- Frequência de palavras-chave  

---

### **4. Geração automática de PDF profissional**

O relatório inclui:

- Capa com thumbnail do vídeo  
- Resumo geral dos comentários  
- Tabelas estatísticas  
- Gráfico de distribuição de contextos  
- Nuvem de palavras (WordCloud)  
- Lista completa e analisada dos comentários  
- Data e metadados  

Arquivos são salvos em:
---
youtube_comments/<VIDEO_ID>/
---

### **5. Suporte a múltiplos vídeos**

O arquivo principal permite processar uma lista:

```python
VIDEO_IDS = [
    "8xg3vE8Ie_E",
    "TAqZb52sgpU",
    "AkFqg5wAuFk"
]
```
Cada vídeo gera sua própria pasta de saída.

## **🧠 Arquitetura de NLP**

Modelos utilizados:
---
Groq Llama-3.1-8B-Instant (opcional)
OpenAI GPT-4.1-mini
---
Tasks realizadas por LLM:

- Sentimento
- Emoção
- Contexto
- Tradução inteligente
- Palavras-chave
- Resumo consolidado

## **📦 Estrutura do Projeto**

```bash
Projeto-NLP-UFABC/
│
├── teste.py                     # Script principal da pipeline NLP
├── requirements.txt             # Dependências do projeto
├── .env                         # Variáveis de ambiente (não subir para o GitHub)
├── 2025_Q3_PLN_PROJETO_PRÁTICO.ipynb   # Notebook de desenvolvimento
│
└── youtube_comments/
     ├── <VIDEO_ID>/
     │    ├── comentarios_youtube_<id>.json
     │    ├── comentarios_analisados_<id>.json
     │    ├── stats_resumo_<id>.json
     │    ├── thumbnail_<id>.jpg
     │    ├── wordcloud_<id>.png
     │    ├── context_chart_<id>.png
     │    └── relatorio_<id>.pdf
```

## **🔧 Instalação**

1. Clone o repositório
```bash
git clone git@github.com:henrique1606/Projeto-NLP-UFABC.git
cd Projeto-NLP-UFABC
```

2. Crie um ambiente virtual
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Instalar as dependencias
```bash
pip install -r requirements.txt
```

## **🔐 Configuração das chaves de API**

Crie um arquivo .env na raiz do projeto:
```bash
touch .env
```

Coloque dentro dele:
```python
    YOUTUBE_API_KEY=SEU_TOKEN_AQUI
    GROQ_API_KEY=SEU_TOKEN_AQUI
    OPENAI_API_KEY=SEU_TOKEN_AQUI
```
Observação:
O arquivo .env não deve ser enviado ao GitHub, pois contém segredos.
Ele já está presente no .gitignore.


### **▶️ Como Executar**

Com o ambiente ativado e o .env configurado:

```bash
    python teste.py
 ```   

## **📊 Exemplo de Relatório Gerado (PDF)**

Cada PDF contém:

- Thumbnail do vídeo
- Link clicável para o vídeo
- Ordem usada na coleta (relevance, time ou rating)
- Resumo consolidado dos comentários
- Tabela completa de estatísticas
- Wordcloud (nuvem de palavras)
- Gráfico de distribuição de contextos
- Lista detalhada de todos os comentários analisados

## **🧠 Tecnologias utilizadas**

APIs

- YouTube Data API v3
- Groq Llama 3.1 
- OpenAI GPT-4.1-mini 

Bibliotecas principais

- langchain
- google-api-python-client
- pandas
- matplotlib
- wordcloud
- reportlab

## **🛠️ Melhorias futuras**

- Deploy como API FastAPI
- Dashboard interativo com Streamlit
- Suporte ampliado para análise de sentimentos multilíngue
- Detecção automática de spam nos comentários
- Análise temporal (como os comentários evoluem ao longo do tempo)

## **📄 Licença**

Este projeto é livre para uso acadêmico.
Créditos: Henrique Cândido · UFABC