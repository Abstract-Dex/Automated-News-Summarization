import os
import json
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
            input_variables=["title", "body", "link"],
            template="""
           Summarize the following news article as a JSON object with these exact keys:
            - "headline": The original article headline
            - "summary": 2-3 paragraphs summarizing the main points
            - "key_points": An array of 5 key takeaways from the article
            
            Return ONLY the JSON object and nothing else.
            
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

        # Find first { and last } to extract JSON
        json_match = re.search(r'\{.*\}', cleaned_content, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)

            try:
                # Parse JSON
                parsed_json = json.loads(json_str)

                # Build TTS-friendly text by concatenating headline, summary and points
                tts_parts = []

                if "headline" in parsed_json:
                    tts_parts.append(parsed_json["headline"])

                if "summary" in parsed_json:
                    tts_parts.append(parsed_json["summary"])

                if "key_points" in parsed_json and isinstance(parsed_json["key_points"], list):
                    # Join key points into sentences
                    points_text = ". ".join(parsed_json["key_points"])
                    if not points_text.endswith("."):
                        points_text += "."
                    tts_parts.append(points_text)

                # Join all parts with spaces
                cleaned_text = " ".join(tts_parts)

                # Return both original JSON and cleaned text
                return {
                    "summary": cleaned_content,
                    "cleaned_summary": cleaned_text
                }
            except json.JSONDecodeError:
                # If JSON parsing fails, fall back to the original content
                return {
                    "summary": cleaned_content,
                    "cleaned_summary": cleaned_content
                }
        else:
            # If no JSON found, return original content
            return {
                "summary": cleaned_content,
                "cleaned_summary": cleaned_content
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
            You are an expert translator specialized in maintaining the tone, context, and nuances 
            of the original text while providing accurate translations.
            Translate the following news article to {target_lang}.
            
            Return ONLY a JSON object with these exact keys:
            - "translated_headline": The translated headline
            - "translated_body": The translated article text
            
            Headline: {headline}
            Article: {text}
            Target Language: {target_lang}
            """
        )

        # Generate translation
        chain = prompt | translator
        result = chain.invoke({
            "headline": request.headline,
            "text": request.text,
            "target_lang": request.target_lang
        })

        return {"translated_text": result.content}

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Translation failed: {str(e)}")


# TODO: Cleaned summary is cleaning the entire summary. Fix that
