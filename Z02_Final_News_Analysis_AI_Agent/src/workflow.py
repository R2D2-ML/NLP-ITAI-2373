import os
from pathlib import Path
import json
import joblib
from langgraph.graph import StateGraph, END, START
from langchain_openai import ChatOpenAI
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.caches import InMemoryCache
from langchain_core.globals import set_llm_cache
from langchain_community.document_loaders import WebBaseLoader
from src.analysis.newsbot import NewsBotIntelligenceSystem
from nltk.sentiment import SentimentIntensityAnalyzer
from src.utils.global_ import TFIDF_VECTORIZER
from src.utils.global_ import clear_terminal
from src.analysis.topic_modeler import LDA
from .models import UserState, ArticleInsights
from .prompts import NewsPrompts


class Workflow:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1)
        self.prompts = NewsPrompts()
        self.search = GoogleSearchAPIWrapper()
        self.workflow = self._build_workflow()
        self.cache = InMemoryCache()
        self.terminate = False
        set_llm_cache(self.cache)

    # Create LangGraph workflow
    def _build_workflow(self):
        graph = StateGraph(UserState)
        graph.add_node("check_input", self.input_validation)
        graph.add_node("grab_news", self.search_engine)
        graph.add_node("start_conversation", self.newsbot_convo)
        # graph.add_node("insights", self.prompt_user)
        graph.add_edge(START, "grab_news")
        graph.add_conditional_edges("grab_news", self.input_validation)
        # graph.add_edge("", "")
        graph.add_edge("start_conversation", END)
        # graph.add_edge("insights", END)
        return graph.compile()
    
    # Check input for inappropriate or harmful requests
    def input_validation(self, state: UserState):
        """
        Check the user's query for inappropriate or harmful content

        Args:
            state: The agent's saved information.

        Returns:
            String: The next node the agent should use.
        """
        messages = [
            SystemMessage(content=self.prompts.INPUT_VALIDATION_SYSTEM),
            HumanMessage(content=self.prompts.input_validation_user(state.query))
        ]
        response = self.llm.invoke(messages)

        # Check if the LLM considers the query to have inappropriate or harmful requests
        if response.content == "True":
            print("This prompt is innappropriate or contains a harmful request.")
            return "END"
        return "start_conversation"
    
    def search_engine(self, state: UserState):
        messages = [
            SystemMessage(content=self.prompts.SEARCH_QUERY_OPTIMIZER),
            HumanMessage(content=self.prompts.user_search_query_optimizer(state.query))
        ]
        response = self.llm.invoke(messages)

        articles = self.search.results(response.content, num_results=1)

        # for r in articles:
        url = articles[0]["link"]
        print(f"Fetching: {articles[0]['title']} - {url}")
        
        loader = WebBaseLoader(url)
        docs = loader.load()

        messages = [
            SystemMessage(content=self.prompts.DOCUMENT_SUMMARIZER),
            HumanMessage(content=self.prompts.user_document_summarizer(docs))
        ]
        response = self.llm.invoke(messages)

        article_text = response.content

        return {"articles": articles, "article_choices": article_text}
    
    def newsbot_convo(self, state: UserState):
        articles = []

        list_choices = state.article_choices.split("\n")
        
        new_choices = []
        for choice in list_choices:
            new_choices.append(choice[3:])

        cleaned_data = [article.replace('*', '') for article in new_choices]

        clean_data = [article for article in cleaned_data if article]

        for data in clean_data:
            temp_dict = {}
            key_value = data.split(":")
            
            temp_dict["title"] = key_value[0]
            temp_dict["content"] = key_value[1]

            articles.append(temp_dict)

        MODEL = joblib.load('data/models/best_classifier.pkl')

        newsbot = NewsBotIntelligenceSystem(classifier=MODEL,
                                            vectorizer=TFIDF_VECTORIZER,
                                            topic_model=LDA # Pass the trained LDA model
                                            )
        
        article_insights = []
        article_insights_iter = []
        for idx, article in enumerate(articles):
            article_dict = {}
            article_dict[str(idx)] = {}
            insights = newsbot.process_article(article["title"], article["content"])

            choice = ArticleInsights()

            choice.title = f"📰 Title: {insights['title']}"
            article_dict[str(idx)]["title"] = f"📰 Title: {insights['title']}"
            choice.content = f"📝 Content: {insights['content']}"
            article_dict[str(idx)]["content"] = f"📝 Content: {insights['content']}"
            choice.predicted_category = f"\n🏷️ Predicted Category: {insights['predicted_category']} ({insights['category_confidence']:.2%} confidence)"
            article_dict[str(idx)]["predicted_category"] = f"\n🏷️ Predicted Category: {insights['predicted_category']} ({insights['category_confidence']:.2%} confidence)"

            probs = []
            for cat, prob in sorted(insights['category_probabilities'].items(), key=lambda x: x[1], reverse=True):
                probs.append(f"  {cat}: {prob:.3f}")
            
            choice.category_probabilities = probs
            article_dict[str(idx)]["category_probabilities"] = probs
            choice.sentiment = f"\n😊 Sentiment: {insights['sentiment']['sentiment_label']} (score: {insights['sentiment']['compound']:.3f})"
            article_dict[str(idx)]["sentiment"] = f"\n😊 Sentiment: {insights['sentiment']['sentiment_label']} (score: {insights['sentiment']['compound']:.3f})"

            entities = []
            if insights['entities']:
                for entity in insights['entities'][:5]:  # Show first 5
                    entities.append(f"  {entity['text']} ({entity['label']}) - {entity['description']}")
                choice.found_entities = entities
                article_dict[str(idx)]["found_entities"] = entities
            
            overview = []
            for insight in insights['insights']:
                overview.append(f"  {insight}")
            choice.overview = overview
            article_dict[str(idx)]["overview"] = overview

            article_insights.append(choice)
            article_insights_iter.append(article_dict)


        while self.terminate == False:
            clear_terminal()
            print(state.article_choices)
            choice = int(input("\nWhich Article would you like to analyze? (input a number): ")) - 1

            article_state = article_insights_iter[choice][str(choice)]

            clear_terminal()
            print(f"Ok lets analyze article {choice}...\n")
            print(f"Title: {article_state["title"]}")
            print(f"Content: {article_state["content"]}\n")
            while True:
                print(f"""
                        What insights would you like to see?
                            1. Predicted Category
                            2. Category Probabilities
                            3. Sentiment
                            4. Found Entities
                            5. Overview
                            6. Change Article
                            7. New Search Query
                    """)
                choice = int(input(""))
                clear_terminal()
                if choice == 7:
                    self.terminate = True
                    clear_terminal()
                    break
                elif choice == 1:
                    print(f"{article_state["predicted_category"]}\n")
                elif choice == 2:
                    print(f"{article_state["category_probabilities"]}\n")
                elif choice == 3:
                    print(f"{article_state["sentiment"]}\n")
                elif choice == 4:
                    print(f"{article_state["found_entities"]}\n")
                elif choice == 5:
                    print(f"{article_state["overview"]}")
                elif choice == 6:
                    clear_terminal()
                    break

        return {}
    
    def prompt_user(self, state: UserState):
        pass
        

            
    

    def run(self, query: str) -> UserState:
        """
        Run the agent

        Args:
            topic: The users query to be searched.

        Returns:
            Object: The final agent state
        """
        starting_state = UserState(query=query)
        finished_state = self.workflow.invoke(starting_state)
        return UserState(**finished_state)