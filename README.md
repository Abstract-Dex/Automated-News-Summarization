# News Summary App

This is a simple news summary app that uses a News API to fetch news articles and a Large Language Model (LLM) to summarize articles.
The app uses `Kokoro Text-to-speech` model from `HuggingFace` to convert news articles into speech for the user to listen to. It supports multiple languages and can be configured to read the summaries in the user's preferred language.

## Features

- Fetch news articles from the News APIs
  - Customizable search query and filters
  - Customizable news source and category
  - Customizable time range
- Summarize the articles using a Large Language Model (LLM)
- Display the summarized articles in a simple app interface
- Option to translate the articles to a different language and read them in the translated language

## Files

- Under `API/src/`:

  - `main.py`: Contains the main application logic, including fetching news articles, summarizing them, and handling user requests.
  - `Dockerfile`: Defines the Docker image for the application.
  - `requirements.txt`: Lists the Python dependencies required for the application.

- Under `utils/`:
  - `tts.py`: Contains functions to convert text to speech using the Kokoro Text-to-speech model.
