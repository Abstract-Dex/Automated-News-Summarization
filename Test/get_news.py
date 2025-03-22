import requests
import os
from typing import List, Optional, Dict, Any
from datetime import date
from pydantic import BaseModel, Field, HttpUrl, field_serializer
from dotenv import load_dotenv

# Load API key from environment variables
load_dotenv()
WORLDNEWS_API_KEY = os.getenv("WORLDNEWS_API_KEY")

# Base URL for the API
BASE_URL = "https://api.worldnewsapi.com/search-news"

# Pydantic models for request and response


class NewsQueryParams(BaseModel):
    text: str = Field(description="Search query text")
    language: str = Field("en", description="Language of news articles")
    source_country: str = Field(
        "in", alias="source-country", description="Country source of news")
    earliest_publish_date: Optional[date] = Field(
        None, alias="earliest-publish-date", description="Earliest publication date")

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True
    }

    @field_serializer('earliest_publish_date')
    def serialize_date(self, dt: Optional[date]) -> Optional[str]:
        if dt is not None:
            return dt.isoformat()
        return None


class NewsArticle(BaseModel):
    id: int
    title: str
    url: HttpUrl
    image_url: Optional[HttpUrl] = Field(None, alias="image")
    publish_date: str = Field(alias="publish_date")
    text: str
    author: Optional[str] = None
    source: Optional[str] = None
    summary: Optional[str] = None

    model_config = {
        "populate_by_name": True
    }


class NewsResponse(BaseModel):
    news: List[NewsArticle]
    offset: int
    number: int
    available: int


class WorldNewsAPI:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.headers = {"x-api-key": api_key}

    def get_news(self, query_params: NewsQueryParams) -> NewsResponse:
        """
        Fetch news articles based on the provided query parameters

        Args:
            query_params: NewsQueryParams object with search criteria

        Returns:
            NewsResponse object containing news articles and metadata

        Raises:
            ValueError: If the API request fails
        """
        # Convert pydantic model to dict for request params (using model_dump instead of dict())
        params = query_params.model_dump(by_alias=True, exclude_none=True)

        # Make API request
        response = requests.get(BASE_URL, headers=self.headers, params=params)

        # Check if request was successful
        if response.status_code == 200:
            # Parse response data through pydantic model (using model_validate instead of parse_obj)
            return NewsResponse.model_validate(response.json())
        else:
            raise ValueError(f"Error {response.status_code}: {response.text}")


# Example usage
if __name__ == "__main__":
    # Create API client
    api = WorldNewsAPI(api_key=WORLDNEWS_API_KEY)

    # Create query parameters
    query = NewsQueryParams(
        text="earthquake",
        language="en",
        source_country="in",
        earliest_publish_date=date(2025, 3, 1)
    )

    try:
        # Get news articles
        results = api.get_news(query)
    except ValueError as e:
        print(e)
