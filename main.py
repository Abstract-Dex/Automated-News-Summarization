from fastapi import FastAPI, HTTPException, Query
import uvicorn
from summarizer import NewsCatcher
from pydantic import BaseModel

app = FastAPI()

news_catcher = NewsCatcher()


class AnswerResponse(BaseModel):
    message: str


@app.get("/")
async def root():
    return {"message": "Automated News Summarization API"}


@app.on_event("startup")
async def startup_event():
    print("Starting up the API")


@app.get("/summarize_news")
@app.post("/summarize_news")
async def summarize_news(topic: str = Query(...), query: str = Query(...), lang: str = Query(...)):
    # keep the lang parameter optional
    # if lang is not provided, set it to "en"
    # if lang is provided, use it
    # else, raise an error
    if lang is None:
        lang = "en"
        print("No language code provided. Defaulting to English.")
    if lang not in ["en", "hi", "bn", "gu", "te", "mr", "ta", "kn", "ml", "pa", "si", "ur", "zu"]:
        raise HTTPException(status_code=400, detail="Invalid language code.")

    topic = topic.lower()
    query = query.lower()
    lang = lang.lower()
    try:
        all_news = news_catcher.fetch_news(topic=topic, query=query)
        if 'articles' not in all_news or not all_news['articles']:
            raise HTTPException(
                status_code=404, detail="No articles found in the response.")
        summaries = news_catcher.summarize(all_news)[:2]
        return {"summaries": summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
