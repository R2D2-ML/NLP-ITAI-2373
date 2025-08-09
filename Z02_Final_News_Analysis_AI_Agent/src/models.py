from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class ChatState(BaseModel):
    chat_end: bool

class ArticleInsights(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    predicted_category: Optional[str] = None
    category_probabilities: Optional[List[str]] = None
    sentiment: Optional[str] = None
    found_entities: Optional[List[str]] = None
    overview: Optional[List[str]] = None

class Article(BaseModel):
    title: str
    content: str

class UserState(BaseModel):
    query: str
    article_choices: Optional[str] = None
    articles: Optional[List[Dict]] = None
    articles_insights: Optional[List[ArticleInsights]] = None
    articles_insights_dict: List[Dict] = Field(default_factory=list)




