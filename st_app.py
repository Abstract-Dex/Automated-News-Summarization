from tts import get_voice
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
import os
import dotenv
import requests
import streamlit as st

from googletrans import Translator      # ameyo: translator
import asyncio                          # ameyo: translator

import soundfile as sf
from IPython.display import display, Audio
from kokoro import KPipeline
import spacy
spacy.load('en_core_web_sm')


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
    A class to fetch news articles using the NewsCatcher API and summarize the content using the LangChain and Groq Inference.

    Attributes:
    api_key: str
        The API key to access the NewsCatcher API.
    llm: ChatGroq
        The LangChain and Groq model to summarize the content of the news articles.

    Methods:
    fetch_news(query, topic, lang, sort_by, page, from_date, to_date, countries):
        Fetch news articles using the NewsCatcher API.
        query: The search query
        topic: The topic of the news articles (Accepted values: news, sport,tech, world, finance,
        politics, business, economics, entertainment, beauty, travel, music, food, science, gaming, energy)
        lang: The language of the news articles (Default: en)
        sort_by: The sorting order of the news articles (Accepted values: relevancy, rank; Default: relevancy)
        page: The page number of the news articles (Default: 1)
        from_date: The start date of the news articles (Default: Yesterday)
        to_date: The end date of the news articles (Default: Today)
        countries: The country code of the news articles (Default: IN)

    summarize(all_news):
        Summarize the content of the news articles using the LangChain and Groq model.
        all_news: The news articles fetched using the NewsCatcher API.
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
        self.summarized_content = ""

        # ameyo: translator
        # self.translator = TranslatorService()
        # self.translator = Translator() ---

    def fetch_news(self, topic, query="*", lang="en", sort_by="relevancy", page=1, from_date="Yesterday", to_date="Today", countries="IN"):
        url = "https://api.newscatcherapi.com/v2/search"
        querystring = {"q": query, "topic": topic, "lang": lang, "sort_by": sort_by,
                       "page": page, "from": from_date, "to": to_date, "countries": countries}
        headers = {"x-api-key": self.api_key}
        response = requests.request(
            "GET", url, headers=headers, params=querystring)
        news = response.json()
        return news

    # ameyo: translator
    def translate(self, text: str, tolang: str) -> str:
        # return asyncio.run(self.translator.translate(text, dest=tolang)).text
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

        if 'articles' not in all_news:
            st.error("No articles found in the response.")
            return

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
                try:
                    index = res.content.index("</think>")
                    res = res.content[index+len("</think>"):]
                except ValueError:
                    pass
                st.write("Setting summary...")
                self.summarized_content = res
                print(res)

                if language_to_iso[language] != "en":
                    with st.spinner(f'Translating to {language}...'):
                        try:
                            translation = self.translate(
                                res, tolang=language_to_iso[lang])
                            # st.subheader(f"Translated Summary ({language}):")
                            st.markdown(translation)
                        except Exception as e:
                            st.error(f"Translation failed: {str(e)}")
                else:
                    st.markdown(res)
                    st.markdown("Source: " + link)
                    st.markdown("---")

            #     # print(res.content)
            #     if language == "en":
            #         st.markdown(res.content)
            #     else:
            #         with st.spinner(f'Translating to {language}...'):
            #             translated_content = self.translator.translate(
            #                 res.content, dest=language)
            #             st.markdown(translated_content)
            #     st.markdown(res.content)
            #     st.markdown("Source: " + link)
            #     st.markdown("---")

            #     translated_content = self.translator.translate(
            #         res.content, dest=lang)
            #     st.markdown(translated_content)
            #     st.markdown("\n\n")
            #     print("\n\n")
            # break

    def generator(self, text, language, gender):
        lang, voice = get_voice(language, gender)
        pipeline = KPipeline(lang_code=lang)
        generator = pipeline(
            text, voice=voice, speed=1, split_pattern=r'\n+')
        for i, (gs, ps, audio) in enumerate(generator):
            sf.write(f'tts_output.wav', audio, 24000)


# def main(topic: str, query: str, language: str):
#     news = NewsCatcher()
#     all_news = news.fetch_news(topic=topic, query=query)
#     summary = news.summarize(all_news, language)
#     return summary

st.title("Automated News Summarization")
topic = st.selectbox("Select the topic", ["news", "sport", "tech", "world", "finance", "politics", "business",
                                          "economics", "entertainment", "beauty", "travel", "music", "food", "science", "gaming", "energy"])
query = st.text_input("Enter the search query")
lang = st.selectbox("Select your language", language_to_iso.keys())
tts_lang = st.selectbox(
    "Select the language for the audio output", ["English", "British English", "Hindi"])
gender = st.selectbox("Select the voice gender", ['Male', 'Female'])

news = NewsCatcher()

summary = None
all_news = news.fetch_news(topic=topic, query=query,
                           lang=language_to_iso[lang])

if st.button("Summarize"):
    summary = news.summarize(all_news, lang)
    st.write(f"Summary in summarize:\n{news.summarized_content}")
    st.markdown(summary)

if st.button("Generate Audio"):
    st.write("Summary in tts: ")
    st.write(news.summarized_content)

    news.generator(news.summarized_content, "English", "Male")
    st.audio("tts_output.wav", format='audio/wav')
    # TODO: Handle cases where user does not provide search query. The search space will automatically be filled with the latest news.
