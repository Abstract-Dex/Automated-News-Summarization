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
async def summarize_news(topic: str = Query(...), query: str = Query(...)):
    try:
        all_news = news_catcher.fetch_news(topic=topic, query=query)
        summaries = news_catcher.summarize(all_news)
        return {"summaries": summaries}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
