from tts import TextToSpeech
from text_cleaning import clean_markdown_for_tts
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
import dotenv
import requests
import streamlit as st

import soundfile as sf

dotenv.load_dotenv()

NEWSCATCHER_API_KEY = os.getenv("NEWSCATCHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

language_to_iso = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Telugu": "te",
    "Marathi": "mr",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Sinhala": "si",
    "Urdu": "ur",
    "Zulu": "zu",
}

tts_languages = {
    "English": "a",
    "British English": "b",
    "Hindi": "h",
}


class NewsCatcher:
    """
    A class to fetch news articles using the NewsCatcher API and summarize the content using LangChain and Groq Inference.
    """

    def __init__(self) -> None:
        self.api_key = NEWSCATCHER_API_KEY
        self.llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model="deepseek-r1-distill-llama-70b",
            temperature=0.0,
            max_retries=2,
            streaming=True,
        )
        self.translator = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model="deepseek-r1-distill-llama-70b",
            temperature=0.0,
            max_retries=2,
            streaming=True,
        )
        # Storage for different content versions
        self.english_markdown = ""
        self.english_clean = ""
        self.translated_markdown = ""
        self.translated_clean = ""
        self.display_language = "English"
        self.tts = TextToSpeech()

    def fetch_news(self, topic, query="*", lang="en", sort_by="relevancy", page=1, from_date="Yesterday", to_date="Today", countries="IN"):
        url = "https://api.newscatcherapi.com/v2/search"
        querystring = {"q": query, "topic": topic, "lang": lang, "sort_by": sort_by,
                       "page": page, "from": from_date, "to": to_date, "countries": countries}
        headers = {"x-api-key": self.api_key}
        response = requests.request(
            "GET", url, headers=headers, params=querystring)
        news = response.json()
        return news

    def translate(self, text: str, tolang: str) -> str:
        prompt = PromptTemplate(
            input_variables=["text", "language"],
            template="""
            Translate the following text to {language}: {text}
            """
        )
        chain = prompt | self.translator
        res = chain.invoke({"text": text, "language": tolang})
        try:
            index = res.content.index("</think>")
            res = res.content[index+len("</think>"):]
        except ValueError:
            pass
        return res

    def summarize(self, all_news, language: str):
        """Summarize news and store both original and translated content"""
        if 'articles' not in all_news or not all_news['articles']:
            st.error("No articles found in the response.")
            return None

        # Update display language
        self.display_language = language

        prompt = PromptTemplate(
            input_variables=["title", "link"],
            template="""
                You are an AI assistant. Your task is to read the news title and visit the provided news link.
                After visiting the link, summarize the content of the news article. Make sure to provide a Heading, Summary and Key Points and finally a Conclusion. Do not mention the subheading preambles.

                News Title: {title}
                News Link: {link}

                Please provide a concise summary of the article in markdown format and exclude the output preamble.

                Also do not output your thought process.
            """
        )

        for info in all_news['articles'][:1]:
            with st.spinner('Generating summary...'):
                title = info['title']
                link = info['link']
                chain = prompt | self.llm
                res = chain.invoke({"title": title, "link": link})

                # Clean the content if needed
                try:
                    index = res.content.index("</think>")
                    cleaned_content = res.content[index+len("</think>"):]
                except ValueError:
                    cleaned_content = res.content

                # Store English versions
                self.english_markdown = cleaned_content
                self.english_clean = clean_markdown_for_tts(cleaned_content)

                # Handle translation if needed
                if language_to_iso[language] != "en":
                    with st.spinner(f'Translating to {language}...'):
                        try:
                            translation = self.translate(
                                cleaned_content, tolang=language_to_iso[language])

                            # Store translated versions
                            self.translated_markdown = translation
                            self.translated_clean = clean_markdown_for_tts(
                                translation)

                            # Display translated content
                            st.markdown(translation)
                        except Exception as e:
                            st.error(f"Translation failed: {str(e)}")
                else:
                    # Clear translation data when in English
                    self.translated_markdown = ""
                    self.translated_clean = ""
                    # Display English content
                    st.markdown(cleaned_content)

                st.markdown("Source: " + link)
                st.markdown("---")

                # Return the displayed content
                return self.translated_markdown if self.translated_markdown else self.english_markdown

    def get_tts_content(self, tts_lang):
        """Get the appropriate content for TTS based on language preference"""
        # Check if we should use the translated content
        if tts_lang in ["Hindi"] and self.translated_clean:
            return self.translated_clean
        # Default to English clean content
        return self.english_clean

    def generate_audio(self, tts_lang: str, gender: str) -> str:
        """Generate audio using the appropriate language content"""
        try:
            # Get the right content based on TTS language
            content_to_use = self.get_tts_content(tts_lang)

            if not content_to_use:
                raise ValueError("No text content available for TTS")

            return self.tts.generate_audio(content_to_use, tts_lang, gender)

        except Exception as e:
            raise RuntimeError(f"Audio generation failed: {str(e)}")


# Streamlit UI
st.title("Automated News Summarization")
topic = st.selectbox("Select the topic", ["news", "sport", "tech", "world", "finance", "politics", "business",
                                          "economics", "entertainment", "beauty", "travel", "music", "food", "science", "gaming", "energy"])
query = st.text_input("Enter the search query")
lang = st.selectbox("Select your language", language_to_iso.keys())
tts_lang = st.selectbox(
    "Select the language for the audio output", ["English", "British English", "Hindi"])
gender = st.selectbox("Select the voice gender", ['Male', 'Female'])

# Initialize the NewsCatcher object
news = NewsCatcher()

# Create a session state to store the content
if 'english_markdown' not in st.session_state:
    st.session_state.english_markdown = ""
if 'english_clean' not in st.session_state:
    st.session_state.english_clean = ""
if 'translated_markdown' not in st.session_state:
    st.session_state.translated_markdown = ""
if 'translated_clean' not in st.session_state:
    st.session_state.translated_clean = ""
if 'display_language' not in st.session_state:
    st.session_state.display_language = "English"

# Fetch news button
if st.button("Summarize"):
    all_news = news.fetch_news(topic=topic, query=query,
                               lang=language_to_iso[lang])
    summary = news.summarize(all_news, lang)
    if summary:
        # Store all content versions in session state
        st.session_state.english_markdown = news.english_markdown
        st.session_state.english_clean = news.english_clean
        st.session_state.translated_markdown = news.translated_markdown
        st.session_state.translated_clean = news.translated_clean
        st.session_state.display_language = news.display_language

# Generate audio button
if st.button("Generate Audio"):
    try:
        # Restore content from session state if needed
        if not news.english_clean and st.session_state.english_clean:
            news.english_markdown = st.session_state.english_markdown
            news.english_clean = st.session_state.english_clean
            news.translated_markdown = st.session_state.translated_markdown
            news.translated_clean = st.session_state.translated_clean
            news.display_language = st.session_state.display_language

        # Determine which content to use based on TTS language
        content_to_use = news.get_tts_content(tts_lang)

        if not content_to_use:
            st.error("Please generate a summary first")
        else:
            with st.spinner("Generating audio..."):
                # Display the clean text that will be used for TTS
                st.info(
                    f"Converting the following text to speech in {tts_lang}:")
                st.text(
                    content_to_use[:200] + "..." if len(content_to_use) > 200 else content_to_use)

                audio_file = news.generate_audio(tts_lang, gender)
                st.audio(audio_file)

    except Exception as e:
        st.error(f"Audio generation failed: {str(e)}")
