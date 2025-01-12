import os
import dotenv
import requests
import streamlit as st

from googletrans import Translator      # ameyo: translator
import asyncio                          # ameyo: translator

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

# from translate import TranslatorService

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
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_retries=2,
            streaming=True,
        )

        # ameyo: translator
        # self.translator = TranslatorService()
        self.translator = Translator()

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
        return asyncio.run(self.translator.translate(text, dest=tolang)).text

    def summarize(self, all_news, language: str):

        if 'articles' not in all_news:
            st.error("No articles found in the response.")
            return

        prompt = PromptTemplate(
            input_variables=["title", "link"],
            template="""
				You are an AI assistant. Your task is to read the news title and visit the provided news link. 
				After visiting the link, summarize the content of the news article.

				News Title: {title}
				News Link: {link}

				Please provide a concise summary of the article in markdown format and exclude the output preamble.
				"""
        )
        for info in all_news['articles'][:1]:
            with st.spinner('Generating summary...'):
                title = info['title']
                link = info['link']
                chain = prompt | self.llm
                res = chain.invoke({"title": title, "link": link})
                # st.markdown("Language: " + language)

                if language != "en":
                    with st.spinner(f'Translating to {language}...'):
                        try:
                            translation = self.translate(res.content, tolang=language_to_iso[lang])
                            st.subheader(f"Translated Summary ({language}):")
                            # Access the text attribute of translation result
                            st.markdown(translation)
                        except Exception as e:
                            st.error(f"Translation failed: {str(e)}")
                else:
                    st.subheader("English Summary:")
                    st.markdown(res.content)

                # print(res.content)
                # if language == "en":
                #     st.markdown(res.content)
                # else:
                #     with st.spinner(f'Translating to {language}...'):
                #         translated_content = self.translator.translate(
                #             res.content, dest=language)
                #         st.markdown(translated_content)
                # st.markdown(res.content)
                st.markdown("Source: " + link)
                st.markdown("---")

                # translated_content = self.translator.translate(
                #     res.content, dest=lang)
                # st.markdown(translated_content)
                # st.markdown("\n\n")
                # print("\n\n")
            # break


def main(topic: str, query: str, language: str):
    news = NewsCatcher()
    all_news = news.fetch_news(topic=topic, query=query)
    news.summarize(all_news, language)


if __name__ == "__main__":
    st.title("Automated News Summarization")
    topic = st.selectbox("Select the topic", ["news", "sport", "tech", "world", "finance", "politics", "business",
                         "economics", "entertainment", "beauty", "travel", "music", "food", "science", "gaming", "energy"])
    query = st.text_input("Enter the search query")
    lang = st.selectbox("Select your language", language_to_iso.keys())
    # lang = st.text_input("Select your language")
    if st.button("Summarize"):
        main(topic=topic, query=query, language=lang)


# TODO: Handle cases where user does not provide search query. The search space will automatically be filled with the latest news.
