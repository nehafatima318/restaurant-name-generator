from langchain_google_genai import ChatGoogleGenerativeAI              
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

llm = ChatGoogleGenerativeAI(
            model="gemini-3.5-flash-lite",
            api_key=api_key,
            timeout=30,
            max_retries=0,
            temperature= 0.6
        )

def generate_restaurant_name_and_items(cuisine):
            # Chain 1: Generate restaurant name
            prompt_template_name = PromptTemplate(
                input_variables=["cuisine"],
                template="Suggest ONE fancy restaurant name for {cuisine} cuisine.Return ONLY the restaurant name.No explanation, no meaning, no tagline, no markdown."
            )

            name_chain = prompt_template_name | llm | StrOutputParser()

            # Chain 2: Generate menu items
            prompt_template_items = PromptTemplate(
                input_variables=["restaurant_name"],
                template="Suggest some menu items for {restaurant_name}. Return them as a comma-separated list."
            )

            food_item_chain = prompt_template_items | llm | StrOutputParser()

            # Sequential chain retaining both outputs
            chain = (
                RunnablePassthrough.assign(
                    restaurant_name=lambda x: name_chain.invoke(x["cuisine"])
                )
                .assign(
                    food_items=lambda x: food_item_chain.invoke(
                        {"restaurant_name": x["restaurant_name"]}
                    )
                )
            )

            # Execute
            response = chain.invoke({"cuisine": cuisine})

            print("Restaurant Name:", response["restaurant_name"])
            print("Food Items:", response["food_items"])

            return response


if __name__ == "__main__":
        print(generate_restaurant_name_and_items("Italian"))