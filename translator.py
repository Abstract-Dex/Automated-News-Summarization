from googletrans import Translator
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

app = FastAPI(title="Translator API")


class TranslatorModel(BaseModel):
    text: str
    dest: str


class ResponseModel(BaseModel):
    text: str


@app.post("/translate", response_model=ResponseModel)
def translate_post(data: TranslatorModel):
    try:
        translator = Translator()
        result = translator.translate(data.text, dest=data.dest)
        return ResponseModel(text=result.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/translate", response_model=ResponseModel)
def translate_get(text: str, dest: str):
    try:
        translator = Translator()
        result = translator.translate(text, dest=dest)
        return ResponseModel(text=result.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)