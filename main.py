from __future__ import annotations
import asyncio
import os
from pydantic import BaseModel, Field
from pydantic.types import SecretStr

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

from maxim import Maxim,Config
from maxim.logger.openai.agents import MaximOpenAIAgentsTracingProcessor

load_dotenv()

from agents import (
    Agent,
    RunContextWrapper,
    Runner,
    add_trace_processor,
    function_tool,
    TResponseInputItem
)

from browser_use import Agent as BrowserAgent, Browser

logger = Maxim(Config()).logger()
add_trace_processor(MaximOpenAIAgentsTracingProcessor(logger))

## CONTEXT

class MobileComparisonContext(BaseModel):
    phoneModel1: str = Field(default="", description="The first phone model provided by the user")
    phoneModel2: str = Field(default="", description="The second phone model provided by the user")
    phoneModel1FeatureData: str = Field(default="", description="Raw data about the first phone model")
    phoneModel2FeatureData: str = Field(default="", description="Raw data about the second phone model")
    comparisonMarkdown: str = Field(default="", description="A structured markdown table comparing both phone models")


## TOOLS

## TOOLS

@function_tool(
    name_override="online_search",
    description_override="Searches online for phone model features and specifications."
)
async def online_search(
    context: RunContextWrapper[MobileComparisonContext], 
    query: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash-exp', api_key=SecretStr(api_key))
    agent = BrowserAgent(
        task=f"Find detailed specifications and features of {query} and return structured data.",
        llm=llm,
        browser=Browser()
    )
    history = await agent.run()
    
    ## update the context with the extracted content
    if context.context.phoneModel1FeatureData == "":
        context.context.phoneModel1FeatureData = history.extracted_content()
    else:
        context.context.phoneModel2FeatureData = history.extracted_content()
    print("updated context - ", context.context)
    return history.extracted_content()




## AGENTS

# 1. Online Search Agent
online_search_agent = Agent[MobileComparisonContext](
    name="Online Search Agent",
    handoff_description="An agent that searches for phone specifications online.",
    instructions="""
    You are an Online Search Agent. Your goal is to fetch detailed specifications for a phone model.
    # Routine:
    1. Use the `online_search` tool to get specifications for the given phone model.
    2. Save the retrieved information in the context.
    """,
    tools=[online_search],
)

# 2. Triage Agent
triage_agent = Agent[MobileComparisonContext](
    name="Triage Agent",
    handoff_description="An agent that delegates tasks to the correct sub-agents.",
    instructions="""
    You are a Triage Agent. Your goal is to coordinate the process of phone model comparison. You have to 
    delegate tasks to the Online Search Agent. 
    # Sequence:
    1. Receive two phone model names from the user.
    2. Handoff the first phone model to the `Online Search Agent`.
    3. Handoff the second phone model to the `Online Search Agent`.
    4. Return the proper comparison in table form in markdown format string.
    
    
    """,
    handoffs=[online_search_agent],
)



## RUNNER

async def main():
    context = MobileComparisonContext()
    phone1 = input("Enter the first phone model 1: ")
    phone2 = input("Enter the second phone model 2: ")
    context.phoneModel1 = phone1
    context.phoneModel2 = phone2
    input_items: list[TResponseInputItem] = [{"content": f"{phone1} vs {phone2}", "role": "user"}]
    result = await Runner.run(triage_agent, input_items, context=context)
    
    print("Final Comparison Markdown:")
    print(result)
    
    # save the final result to a file in same folder
    
    with open("comparison.md", "w") as f:
        f.write(result.final_output)
    print("Comparison saved to comparison.md file")
    
    
if __name__ == "__main__":
    asyncio.run(main())
