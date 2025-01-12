# News Summary App

This is a simple news summary app that uses the [NewsCatcher API](https://newscatcherapi.com/) to fetch news articles and a Large Language Model (LLM) like the Meta Llama to summarize the articles. 

The app uses `Indic Parler Text-to-speech` model from `HuggingFace` to convert news articles into speech for the user to listen to. It supports multiple languages and can be configured to read the summaries in the user's preferred language.

## Features

- Fetch news articles from the NewsCatcher API
  - Customizable search query and filters
  - Customizable news source and category
  - Customizable time range
- Summarize the articles using a Large Language Model (LLM)
- Display the summarized articles in a simple web interface
- Option to translate the articles to a different language and read them in the translated language

## Files

- `summarizer.py`: Contains the summarizer class that uses the LLM to summarize articles
- `main.py`: Contains the main code for the api endpoints
- `st_app.py`: Contains the streamlit app code for the web interface
- `translate.py`: Contains the translator class that uses the Google Translate API to translate articles
- `tts.py`: Contains the text-to-speech class that uses the Indic Parler Text-to-speech model to convert articles into speech

### Note: This is a work in progress and is not yet complete
