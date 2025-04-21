import os
import re
import requests
import uvicorn
from typing import Optional
from mangum import Mangum

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

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
    headline: str
    text: str
    target_lang: str


class TranslatorResponse(BaseModel):
    translated_text: str


llm = ChatGroq(
    model="deepseek-r1-distill-llama-70b",
    api_key=GROQ_API_KEY,
    temperature=0.0,
    max_retries=2,
)


translator = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=GROQ_API_KEY,
    temperature=0.0,
    max_retries=2,
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
            input_variables=["headline", "body", "link"],
            template="""
            <think>
            You are an AI assistant specialized in summarizing news articles.
            Return the summary in JSON format with 'summary' and 'key_points' keys.
            </think>

            Summarize the following news article in this exact JSON format:
            {
              "headline": "article headline here",
              "summary": "2-3 paragraphs summarizing the main points",
              "key_points": ["point 1", "point 2", "point 3", "point 4", "point 5"]
            }

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
            input_variables=["headline", "text", "target_lang"],
            template="""
              <think>
              You are an expert translator specialized in maintaining the tone, context, and nuances 
              of the original text while providing accurate translations.
              Return the translation in JSON format with 'heading' and 'body' keys.
              </think>

              Translate the following news article to {target_lang}, preserving the journalistic style 
              and formal tone. Ensure names, dates, numbers, and technical terms are accurately translated.

              Format your response as:
              {{
            "heading": "translated headline here",
            "body": "translated article text here"
              }}

              Headline: {headline}
              Article: {text}
              Target Language: {target_lang}
              """
        )

        # Generate translation
        chain = prompt | translator
        result = chain.invoke({
            "headline": request.title,
            "text": request.text,
            "target_lang": request.target_lang
        })

        return {"translated_text": result.content}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Translation failed: {str(e)}")
