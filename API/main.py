import os
import re
import dotenv
import requests
import uvicorn
from typing import Optional
from mangum import Mangum

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel


dotenv.load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

app = FastAPI(
    title="Summarizer API",
    description="Summarizer API using Langchain Groq",
    version="1.0.0"
)

handler = Mangum(app)


def clean_markdown_for_tts(markdown_text):
    """Clean markdown syntax for TTS processing"""
    # Remove heading markers (# symbols)
    text = re.sub(r'#+\s+', '', markdown_text)

    # Remove bold/italic markers
    text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)
    text = re.sub(r'\*(.*?)\*', r'\1', text)

    # Remove link syntax but keep link text
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)

    # Remove bullet points but keep text
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE)

    # Replace ordered list markers (1., 2., etc.) with just their content
    text = re.sub(r'^\s*\d+\.\s+', '', text, flags=re.MULTILINE)

    # Remove code blocks and their syntax
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    text = re.sub(r'`(.*?)`', r'\1', text)

    # Remove horizontal rules
    text = re.sub(r'---+', '', text)

    # Remove blockquote markers
    text = re.sub(r'^\s*>\s+', '', text, flags=re.MULTILINE)

    # Normalize multiple newlines to just two
    text = re.sub(r'\n{3,}', '\n\n', text)

    return text


class SummarizerRequest(BaseModel):
    title: str
    body: str
    link: Optional[str] = None


class SummarizerResponse(BaseModel):
    summary: str
    cleaned_summary: str


class TranslatorRequest(BaseModel):
    text: str
    target_lang: str


class TranslatorResponse(BaseModel):
    translated_text: str


llm = ChatGroq(
    model="deepseek-r1-distill-qwen-32b",
    api_key=GROQ_API_KEY,
    temperature=0.0,
    max_retries=2,
    streaming=True,
)


translator = ChatGroq(
    model="gemma2-9b-it",
    api_key=GROQ_API_KEY,
    temperature=0.0,
    max_retries=2,
    streaming=True,
)


@app.get("/")
def read_root():
    return {"message": "Welcome to the Summarizer API"}


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/summarize", response_model=SummarizerResponse)
async def summarize_article(request: SummarizerRequest):
    try:
        prompt = PromptTemplate(
            input_variables=["title", "body", "link"],
            template="""
            <think>
            You are an AI assistant specialized in summarizing news articles.
            </think>

            Summarize the following news article. Provide a:
            1. Summary of the main points (2-3 paragraphs)
            2. Key takeaways (3-5 bullet points)

            Format the response in markdown.

            Title: {title}
            Article: {body}
            Source: {link}
            """
        )

        # Generate summary
        chain = prompt | llm
        result = chain.invoke({
            "title": request.title,
            "body": request.body,
            "link": str(request.link) if request.link else "Not provided"
        })

        # Clean the content
        content = result.content if hasattr(
            result, "content") else str(result)
        try:
            index = content.index("</think>")
            cleaned_content = content[index+len("</think>"):].strip()
        except ValueError:
            cleaned_content = content.strip()

        return {
            "summary": cleaned_content,
            "cleaned_summary": clean_markdown_for_tts(cleaned_content)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Summarization failed: {str(e)}")


@app.post("/translate", response_model=TranslatorResponse)
async def translate_text(request: TranslatorRequest):
    try:
        prompt = PromptTemplate(
            input_variables=["text", "target_lang"],
            template="""
              <think>
              You are an AI assistant specialized in translation. 
              Only return the translated text without any extra words.
              </think>

              Translate the following text to the target language.

              Text: {text}
              Target Language: {target_lang}

              Translation:
              """
        )

        # Generate translation
        chain = prompt | translator
        result = chain.invoke({
            "text": request.text,
            "target_lang": request.target_lang
        })

        return {"translated_text": result.content}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Translation failed: {str(e)}")
