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

🧠 Arquitetura de NLP
Modelos utilizados:

---
Groq Llama-3.1-8B-Instant → rápido e eficiente
OpenAI GPT-4.1-mini (opcional)
---
Tasks realizadas por LLM:

- Sentimento
- Emoção
- Contexto
- Tradução inteligente
- Palavras-chave
- Resumo consolidado