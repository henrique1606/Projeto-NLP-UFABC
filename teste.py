# ============================================
# IMPORTS DO SISTEMA E TERCEIROS
# ============================================
import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================
# GOOGLE API
# ============================================
from googleapiclient.discovery import build

# ============================================
# LANGCHAIN (ATUALIZADO 2025)
# ============================================
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI

# ============================================
# REPORTLAB PARA GERAÇÃO DE PDF
# ============================================
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image
)
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# ============================================
# VISUALIZAÇÃO E NUVEM DE PALAVRAS
# ============================================
from wordcloud import WordCloud


# ============================================================
# CONFIGURAÇÃO DAS CHAVES DE API
# ============================================================

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ============================================================
# CONFIGURAÇÃO DO LLM (Groq — modelos gratuitos)
# ============================================================

def get_llm(provider="groq"):
    """
    Retorna o LLM escolhido.
    - provider="groq" → modelo Groq (padrão)
    - provider="openai" → modelo OpenAI Mini 1
    """

    if provider == "openai":
        return ChatOpenAI(
            model="gpt-4.1-mini",   
            temperature=0.1
        )

    # Groq (padrão)
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1
    )

parser = StrOutputParser()


# ============================================================
# FUNÇÃO: Converter emojis em palavras
# ============================================================

EMOJI_MAP = {
    # ALEGRIA / FELICIDADE
    "😂": "risada", "🤣": "risada", "😅": "risada nervosa", "😁": "feliz",
    "😄": "feliz", "😊": "feliz", "😃": "feliz", "🙂": "feliz leve",
    "😆": "muita risada", "😎": "confiante", "🤩": "empolgado", "🥳": "celebrando",
    "😺": "feliz",

    # AMOR / CARINHO
    "😍": "amor", "🥰": "carinho", "😘": "beijo", "😗": "beijo leve",
    "😻": "amor", "❤️": "amor", "💓": "amor", "💗": "amor",
    "💖": "amor", "💘": "amor", "💝": "carinho", "💕": "carinho",
    "💞": "carinho", "💟": "afeto", "💌": "amor", "💙": "amor",
    "💚": "amor", "💛": "amor", "💜": "amor", "🧡": "amor",
    "🩷": "amor", "✨": "brilho", "🤗": "abraço",

    # TRISTEZA
    "😢": "triste", "😭": "chorando", "🥺": "suplica", "☹️": "triste",
    "😞": "desapontado", "😔": "triste", "😟": "preocupado",

    # RAIVA
    "😡": "raiva", "😠": "raiva", "🤬": "muita raiva",

    # SURPRESA
    "😱": "surpresa", "😮": "surpresa", "😯": "surpresa", "😲": "surpresa",

    # MEDO / ANSIEDADE
    "😨": "medo", "😰": "ansiedade", "😥": "angustia", "🥹": "emoção forte",

    # NOJO
    "🤢": "nojo", "🤮": "nojo extremo",

    # NEUTRO / PENSATIVO
    "😐": "neutro", "😑": "neutro", "🤔": "pensativo", "🤨": "duvida",

    # MÃOS / GESTOS
    "🙏": "gratidão", "👍": "positivo", "👎": "negativo", "👏": "aplausos",
    "🙌": "celebração", "🤝": "parceria", "✌️": "paz", "👌": "ok",
    "🤲": "oferta",

    # ANIMAIS
    "🐶": "cachorro", "🐕": "cachorro", "🐱": "gato", "🐈": "gato",
    "🐾": "patinhas",

    # FOGO / FESTA / MÚSICA
    "🔥": "incrível", "🎉": "festa", "🎊": "celebração", "🎵": "musica",
    "🎶": "musica", "🎤": "microfone", "🎧": "audio", "🎼": "melodia",

    # OUTROS
    "💀": "chocado", "🤡": "palhaçada", "🌟": "estrela", "⭐": "estrela",
    "💥": "impacto",
}


def emoji_to_text(text: str) -> str:
    for emoji, word in EMOJI_MAP.items():
        text = text.replace(emoji, f" {word} ")
    return text.strip()


# ============================================================
# PROMPTS — Funções de PLN
# ============================================================

def build_chains(llm_model):

    # Sentimento
    sentiment_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
                Classifique o sentimento predominante expresso no comentário abaixo,
                considerando que se trata de um comentário sobre uma música.
                Avalie o tom geral da mensagem, a intenção emocional do autor e o impacto implícito,
                incluindo possíveis indicações dadas por emojis ou expressões afetivas.

                Escolha APENAS uma das três opções:
                - positivo   (elogios, carinho, satisfação, emoção boa, nostalgia afetiva)
                - negativo   (críticas, frustração, incômodo, rejeição, emoção ruim)
                - neutro     (informativo, ambíguo ou sem carga emocional clara)

                Comentário:
                {text}

                Regra: responda APENAS com uma das palavras acima, sem explicações adicionais.
            """
        )

    emotion_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
            Classifique a emoção dominante expressa no comentário abaixo.
            Considere o contexto de comentários sobre músicas, incluindo reações à melodia,
            letra, voz, memória afetiva, nostalgia e sentimentos sugeridos por emojis
            já convertidos em texto.

            A resposta deve ser EXATAMENTE uma das emoções da lista principal:

            alegria
            amor
            nostalgia
            saudade
            tristeza
            melancolia
            raiva
            surpresa
            inspiração
            reflexão
            neutro

            Regras obrigatórias:
            - escolha somente UMA palavra da lista;
            - não use frases, justificativas ou variações;
            - não invente emoções fora da lista.

            Comentário:
            {text}

            Resposta:
        """
    )

    # Keywords
    keywords_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
            Extraia entre **5 e 10 palavras-chave realmente significativas** do comentário abaixo.

            O objetivo é identificar elementos centrais do comentário, considerando que ele trata de uma música. Portanto, priorize palavras relacionadas a:

            - emoções (ex: nostalgia, alegria, tristeza)
            - temas mencionados (ex: saudade, lembrança, superação)
            - experiência pessoal (ex: infância, momento, vida)
            - elementos musicais (ex: melodia, voz, letra, ritmo, guitarra)
            - apreciação ou crítica (ex: incrível, poderoso, marcante)
            - impacto afetivo ou sensorial (ex: arrepio, energia, vibe)

            ============================================================
            REGRAS OBRIGATÓRIAS (SIGA À RISCA):
            ============================================================

            1) Não extraia palavras genéricas demais  
            (ex.: música, vídeo, coisa, muito, bom, legal, aí).

            2) NÃO incluir:
            - artigos, pronomes ou conectivos (o, a, que, de, com…)
            - emojis
            - números
            - repetição da mesma palavra
            - trechos longos ou frases inteiras
            - palavras sem valor semântico real

            3) Extraia apenas palavras isoladas (1 palavra cada),  
            sempre no **singular**, sem hashtags.

            4) As palavras-chave devem capturar O ESSENCIAL do comentário:
            — emoções centrais  
            — temas mencionados  
            — experiência afetiva  
            — elementos musicais

            5) Responda SOMENTE com as palavras-chave,  
            separadas por vírgula, sem comentários extras.

            ============================================================
            Comentário:
            {text}

            Responda SOMENTE com as palavras-chave (5 a 10 termos):
        """
    )


    # Resumo final (para todos os comentários)
    summary_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
            Gere um resumo claro, objetivo e bem estruturado sobre o conjunto de comentários abaixo,
            considerando especificamente o contexto de comentários sobre músicas. 
            Leve em conta que usuários costumam expressar emoções intensas, memórias pessoais,
            sensações despertadas pela melodia ou pela letra, identificação com o artista,
            e reações afetivas típicas desse ambiente.

            Sua análise deve identificar:

            - principais opiniões e percepções dos usuários sobre a música, letra, melodia, artista ou impacto emocional;
            - emoções predominantes e padrões emocionais recorrentes (ex.: nostalgia, saudade, alegria, comoção);
            - tendências gerais de sentimento (positivo, negativo ou neutro) relacionadas à experiência musical;
            - temas centrais mencionados, como lembranças, relacionamentos, fases da vida, performance do artista, qualidade da produção ou significado pessoal;
            - contrastes relevantes entre grupos de comentários (ex.: fãs antigos vs. novos ouvintes, experiências pessoais diferentes).

            Use linguagem direta, síntese precisa e foco nas informações realmente relevantes.

            Texto analisado:
            {text}

            Retorne UM ÚNICO parágrafo de até 10 linhas, sem listas e evitando repetições.
        """
    )

    # Classificação do contexto (tipo de relação com a música)
    context_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
            Classifique o COMPORTAMENTO do comentário em relação à música do vídeo.

            A classificação deve ser EXCLUSIVA — escolha apenas UMA opção — e considerar o foco principal do que a pessoa escreveu.

            ======================================================
            CATEGORIAS PERMITIDAS (escolha SOMENTE uma):
            ======================================================

            1) **sobre_a_musica**
            Quando o comentário fala diretamente sobre:
            - a música, letra, melodia, ritmo, harmonia;
            - a performance do artista;
            - opinião, crítica ou elogio sobre o som;
            - produção musical, qualidade do áudio, clipe.

            Exemplos:
            - "Essa música é perfeita!"
            - "O refrão é muito forte."
            - "O vocal dele tá incrível."

            2) **experiencia_pessoal**
            Quando o comentário relata uma memória, história ou situação da vida relacionada à música.

            Exemplos:
            - "Essa música marcou minha adolescência."
            - "Ouvi essa música no meu casamento."
            - "Me lembra meu pai que já faleceu."

            3) **trecho_de_letra**
            Quando o comentário contém um trecho da música, mesmo que modificado levemente.
            Não importa se a pessoa não cita que é letra — identifique pelo conteúdo.

            Exemplos:
            - "I've given up, I'm sick of feeling!"
            - "Walk on home boy!"
            - "Só as antigas vão lembrar…"

            4) **off_topic**
            Quando o comentário NÃO tem relação com a música.
            Inclui:
            - memes aleatórios;
            - política, religião, futebol;
            - conversa paralela com outros usuários;
            - perguntas nada a ver;
            - emojis sem contexto;
            - spam.

            Exemplos:
            - "Quem mais veio por causa do TikTok?"
            - "Brasil 7x1 Alemanha."
            - "Alguém sabe qual é o nome do cachorro?"

            ======================================================
            REGRAS OBRIGATÓRIAS
            ======================================================

            - Escolha APENAS UMA opção.
            - NÃO explique sua escolha, NÃO adicione texto extra.
            - Se houver mistura de elementos, escolha o TEMA PRINCIPAL do comentário.
            - Se o comentário tiver letra + opinião → classifique como **trecho_de_letra**.
            - Se o comentário for só emojis:  
                - Se forem claramente emocionais → classifique como **sobre_a_musica**.  
                - Se forem aleatórios → **off_topic**.

            ======================================================
            Comentário a classificar:
            {text}

            Responda SOMENTE com uma das opções:
            sobre_a_musica, experiencia_pessoal, trecho_de_letra, off_topic
    """
    )

    # Detecção de idioma (melhorada para contexto de música)
    language_prompt = PromptTemplate(
        input_variables=["text"],
        template="""
            Identifique o idioma principal do comentário abaixo.

            IMPORTANTE:
            - Considere que muitos comentários de YouTube sobre músicas podem conter:
                * trechos da letra,
                * nomes de artistas,
                * palavras repetidas,
                * expressões informais,
                * gírias multilíngues,
                * emojis.
            - Nesses casos, identifique o idioma predominante da frase como um todo.
            - Se houver mistura, escolha o idioma da maior parte do texto.

            Responda SOMENTE com o código ISO-639-1:
            - pt, en, es, fr, de, it, etc.

            Comentário:
            {text}

            Responda exclusivamente com o código do idioma, sem frases adicionais.
        """
    )

    # Tradução automática (especializada para comentários de música)
    translate_prompt = PromptTemplate(
        input_variables=["text", "lang"],
        template="""
            Você receberá um comentário de YouTube sobre uma música, junto com o idioma detectado.

            Se o idioma for "pt":
                - NÃO traduza.
                - NÃO reescreva.
                - Retorne o texto exatamente como está.

            Caso contrário:
                - Traduza o comentário para o português brasileiro.
                - Mantenha o sentido original, o tom emocional e o estilo do autor.
                - Preserve:
                    * gírias
                    * expressões culturais
                    * nomes próprios
                    * termos musicais (chorus, beat, flow, vocals, harmony)
                    * trechos de letra de música (sem adaptar)
                - Se houver palavras de vários idiomas no mesmo comentário,
                traduza apenas o que for do idioma detectado como predominante.

            Idioma detectado: {lang}
            Comentário original:
            {text}

            Responda somente com o texto final traduzido ou preservado.
        """
    )

    return {
        "sentiment": sentiment_prompt | llm_model | parser,
        "emotion": emotion_prompt | llm_model | parser,
        "keywords": keywords_prompt | llm_model | parser,
        "summary": summary_prompt | llm_model | parser,
        "context": context_prompt | llm_model | parser,
        "language": language_prompt | llm_model | parser,
        "translate": translate_prompt | llm_model | parser,
    }

LLM_MODEL = get_llm(provider="openai")
CHAINS = build_chains(LLM_MODEL)


# ============================================================
# COLETA DE COMENTÁRIOS DO YOUTUBE
# ============================================================

def extract_youtube_comments(
    video_id: str, 
    max_comments: int = 50, 
    order: str = "relevance"
):

    youtube = build("youtube", "v3", developerKey=os.getenv("YOUTUBE_API_KEY"))
    comments = []

    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        textFormat="plainText",
        maxResults=100,
        order=order  
    )

    response = request.execute()

    while response and len(comments) < max_comments:

        for item in response.get("items", []):
            snippet = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "comment_id": item["snippet"]["topLevelComment"]["id"],
                "author": snippet.get("authorDisplayName"),
                "text": snippet.get("textDisplay", ""),
                "published_at": snippet.get("publishedAt"),
                "like_count": snippet.get("likeCount", 0),
                "comment_url": (
                    f"https://www.youtube.com/watch?v={video_id}&lc="
                    f"{item['snippet']['topLevelComment']['id']}"
                ),
                "video_id": video_id,
            })

            if len(comments) >= max_comments:
                break

        if "nextPageToken" not in response:
            break

        response = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            textFormat="plainText",
            maxResults=100,
            pageToken=response["nextPageToken"],
            order=order
        ).execute()

    return comments, order

# ============================================================
# PROCESSAMENTO INDIVIDUAL
# ============================================================

def process_comment(text: str):

    text_expanded = emoji_to_text(text)

    # 1) Detectar idioma
    lang = CHAINS["language"].invoke({"text": text_expanded}).strip().lower()

    # 2) Traduzir caso não seja PT
    translated = CHAINS["translate"].invoke({
        "text": text_expanded,
        "lang": lang
    }).strip()

    return {
        "emoji_expanded": text_expanded,
        "language": lang,
        "translated": translated,
        "sentiment": CHAINS["sentiment"].invoke({"text": translated}).strip().lower(),
        "emotion": CHAINS["emotion"].invoke({"text": translated}).strip().lower(),
        "keywords": CHAINS["keywords"].invoke({"text": translated}).strip(),
        "context": CHAINS["context"].invoke({"text": translated}).strip().lower()
    }


# ============================================================
# ANÁLISE COMPLETA COM LOGS
# ============================================================

def analyze_comments(comments):
    enriched_data = []

    print("\n=== INICIANDO ANÁLISE DOS COMENTÁRIOS ===\n")

    for idx, c in enumerate(comments, start=1):

        # NOVO: proteção contra formatos inválidos
        if not isinstance(c, dict) or "text" not in c:
            print(f"⚠ Comentário ignorado (formato inesperado): {c}")
            continue

        processed = process_comment(c["text"])
        enriched = {**c, **processed}
        enriched_data.append(enriched)

        print(f"[{idx}/{len(comments)}] Comentário analisado:")
        print(f"Texto: {c['text'][:90]}")
        print(f"Sentimento: {processed['sentiment']} | Emoção: {processed['emotion']} | Contexto: {processed['context']}")
        print("-" * 70)

    print("\n=== ANÁLISE COMPLETA ===\n")

    return enriched_data

# ============================================================
# RESUMO FINAL
# ============================================================

def generate_final_summary(analyzed_comments):
    combined = "\n".join([c["translated"] for c in analyzed_comments])
    return CHAINS["summary"].invoke({"text": combined}).strip()

# ============================================================
# ESTATÍSTICAS
# ============================================================

def generate_stats(analyzed_comments):

    df = pd.DataFrame(analyzed_comments)

    stats = {
        "sentiment_counts": df["sentiment"].value_counts().to_dict() if "sentiment" in df else {},
        "emotion_counts": df["emotion"].value_counts().to_dict() if "emotion" in df else {},
        "context_counts": df["context"].value_counts().to_dict() if "context" in df else {},
        "language_counts": df["language"].value_counts().to_dict() if "language" in df else {}
    }

    # retorna somente o dicionário de stats (não grava nada no diretório raiz)
    return stats


# ============================================================
# SALVAMENTO
# ============================================================

def save_json(filename, data):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def save_outputs_for_video(video_id, comments, analyzed, resumo, stats):
    base_dir = f"youtube_comments/{video_id}"
    os.makedirs(base_dir, exist_ok=True)

    # Arquivos JSON essenciais
    save_json(f"{base_dir}/comentarios_youtube_{video_id}.json", comments)
    save_json(f"{base_dir}/comentarios_analisados_{video_id}.json", analyzed)

    # Estatísticas (mantém JSON)
    with open(f"{base_dir}/stats_resumo_{video_id}.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=4, ensure_ascii=False)


def generate_pdf_report(video_id, resumo, stats, analyzed, order_used):
    base_dir = f"youtube_comments/{video_id}"
    os.makedirs(base_dir, exist_ok=True)

    file_path = f"{base_dir}/relatorio_{video_id}.pdf"

    # Document settings
    doc = SimpleDocTemplate(
        file_path,
        pagesize=A4,
        title=f"Relatório de Análise - {video_id}",
        leftMargin=40,
        rightMargin=40,
        topMargin=40,
        bottomMargin=40,
    )

    styles = getSampleStyleSheet()
    story = []

    # =============================
    # THUMBNAIL DO VÍDEO
    # =============================
    story.append(Paragraph("<b>Capa do Vídeo</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))

    thumb_path = f"{base_dir}/thumbnail_{video_id}.jpg"
    thumb_urls = [
        f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg",
        f"https://img.youtube.com/vi/{video_id}/hqdefault.jpg"
    ]

    downloaded = False
    for url in thumb_urls:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200 and len(r.content) > 1000:
                with open(thumb_path, "wb") as img:
                    img.write(r.content)
                downloaded = True
                break
        except:
            pass

    if downloaded:
        try:
            img = Image(thumb_path)
            img._restrictSize(450, 250)
            story.append(img)
            story.append(Spacer(1, 20))
        except Exception as e:
            print(f"⚠ Erro ao inserir thumbnail no PDF: {e}")
    else:
        story.append(Paragraph("<i>Thumbnail não disponível</i>", styles["BodyText"]))
        story.append(Spacer(1, 20))

    # =============================
    # TÍTULO + LINK + ORDEM
    # =============================
    story.append(Paragraph(
        f"<b>Relatório de Análise de Comentários – Vídeo {video_id}</b>",
        styles["Title"]
    ))
    story.append(Spacer(1, 12))

    video_url = f"https://www.youtube.com/watch?v={video_id}"
    story.append(Paragraph(f'<a href="{video_url}">{video_url}</a>', styles["BodyText"]))
    story.append(Spacer(1, 10))

    story.append(Paragraph(
        f"<i>Ordem de coleta dos comentários: <b>{order_used}</b></i>",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 20))

    # =============================
    # RESUMO FINAL
    # =============================
    story.append(Paragraph("<b>Resumo Geral</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))
    story.append(Paragraph(resumo, styles["BodyText"]))
    story.append(Spacer(1, 20))

    # =============================
    # ESTATÍSTICAS
    # =============================
    story.append(Paragraph("<b>Estatísticas de Comentários</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))

    def fmt_counts(d):
        return "—" if not d else ", ".join([f"{k}: {v}" for k, v in d.items()])

    stats_table = [
        ["Métrica", "Distribuição"],
        ["Sentimentos", fmt_counts(stats.get("sentiment_counts", {}))],
        ["Emoções", fmt_counts(stats.get("emotion_counts", {}))],
        ["Contextos", fmt_counts(stats.get("context_counts", {}))],
        ["Idiomas", fmt_counts(stats.get("language_counts", {}))],
    ]

    table = Table(stats_table, colWidths=[130, 350])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#4B5563")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (0, 0), (-1, -1), "LEFT"),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.black),
    ]))

    story.append(table)
    story.append(Spacer(1, 20))

    # =============================
    # GERA WORDCLOUD
    # =============================
    story.append(Paragraph("<b>Nuvem de Palavras (Keywords)</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))

    all_keywords = []
    for c in analyzed:
        if c.get("keywords"):
            for kw in c["keywords"].split(","):
                all_keywords.append(kw.strip())

    wc_text = " ".join(all_keywords) if all_keywords else "vazio"

    wc = WordCloud(width=800, height=400, background_color="white").generate(wc_text)
    wc_path = f"{base_dir}/wordcloud_{video_id}.png"
    wc.to_file(wc_path)

    story.append(Image(wc_path, width=400, height=200))
    story.append(Spacer(1, 20))

    # =============================
    # GRÁFICO DE CONTEXTO
    # =============================
    story.append(Paragraph("<b>Distribuição de Contextos</b>", styles["Heading2"]))
    story.append(Spacer(1, 6))

    context_counts = stats.get("context_counts", {})

    plt.figure(figsize=(6, 3))
    plt.bar(context_counts.keys(), context_counts.values())
    plt.title("Contextos dos Comentários")
    plt.tight_layout()

    ctx_path = f"{base_dir}/context_chart_{video_id}.png"
    plt.savefig(ctx_path)
    plt.close()

    story.append(Image(ctx_path, width=400, height=200))
    story.append(Spacer(1, 20))

    # =============================
    # LISTA COMPLETA DE COMENTÁRIOS
    # =============================
    story.append(Paragraph("<b>Lista Completa de Comentários</b>", styles["Heading2"]))
    story.append(Spacer(1, 10))

    for idx, c in enumerate(analyzed, start=1):
        text = c.get("translated") or c.get("text") or ""
        lang = c.get("language", "")

        block = (
            f"<b>{idx}.</b> "
            f"{f'[{lang}] ' if lang else ''}{text}"
            f"<br/><i>Sentimento:</i> {c.get('sentiment', '—')} | "
            f"<i>Emoção:</i> {c.get('emotion', '—')} | "
            f"<i>Contexto:</i> {c.get('context', '—')}"
        )
        story.append(Paragraph(block, styles["BodyText"]))
        story.append(Spacer(1, 6))

    story.append(Spacer(1, 20))

    # =============================
    # RODAPÉ
    # =============================
    story.append(Paragraph(
        f"<i>Total de comentários analisados: {len(analyzed)}</i>",
        styles["BodyText"]
    ))
    story.append(Spacer(1, 6))

    story.append(Paragraph(
        f"<i>Relatório gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}</i>",
        styles["BodyText"]
    ))

    # FINALIZA PDF
    doc.build(story)
    print(f"📄 PDF gerado: {file_path}")

# ============================================================
# EXECUÇÃO PRINCIPAL
# ============================================================

if __name__ == "__main__":
    import time

    VIDEO_IDS = [
        "8xg3vE8Ie_E",  # Raimundos - Quero ver o Oco
        #"ysWkTSTuUmw", # Charlie Brown Jr. - Samba Makossa (Acústico MTV)
        #"TAqZb52sgpU", # Linkin Park - Given Up
        #"AkFqg5wAuFk", # Pantera - Walk
    ]

    print("\n===============================================")
    print(" INICIANDO PROCESSAMENTO DOS VÍDEOS DO YOUTUBE ")
    print("===============================================\n")

    for vid in VIDEO_IDS:
        inicio_video = time.time()

        print("-----------------------------------------------")

        try:
            # 1) Coleta de comentários
            comments, order_used = extract_youtube_comments(vid, max_comments=30)
            print(f"📥 Comentários coletados: {len(comments)}")

            # Se não houver comentários, pular vídeo
            if len(comments) == 0:
                print("⚠ Nenhum comentário encontrado. Pulando vídeo.")
                continue

            # 2) Análise dos comentários
            analyzed = analyze_comments(comments)
            print("🧠 Análise concluída.")

            # 3) Resumo geral dos comentários
            resumo = generate_final_summary(analyzed)
            print("📝 Resumo gerado.")

            # 4) Estatísticas gerais
            stats = generate_stats(analyzed)
            print("📊 Estatísticas calculadas.")

            # 5) Salvamento em arquivos organizados
            save_outputs_for_video(
                video_id=vid,
                comments=comments,
                analyzed=analyzed,
                resumo=resumo,
                stats=stats
            )
            print("💾 Arquivos salvos.")

            # 6) Gerar PDF consolidado
            generate_pdf_report(
                video_id=vid,
                resumo=resumo,
                stats=stats,
                analyzed=analyzed,
                order_used=order_used
            )
            print("📄 PDF gerado com sucesso.")

        except Exception as e:
            print(f"❌ ERRO ao processar o vídeo {vid}: {e}")
            continue

        fim_video = time.time()
        print(f"⏱ Tempo total: {fim_video - inicio_video:.2f} segundos")
        print(f"✔ Finalizado: youtube_comments/{vid}")
        print("-----------------------------------------------")

    print("\n🎉 PROCESSAMENTO FINALIZADO PARA TODOS OS VÍDEOS!\n")

