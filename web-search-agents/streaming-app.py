from langchain.agents import AgentType, initialize_agent
from langchain.tools import DuckDuckGoSearchRun, Tool
from langchain.llms import OpenAI
import yfinance as yf

def stock_info(ticker):
    stock = yf.Ticker(ticker)
    return stock.info

tools = [
    DuckDuckGoSearchRun(),
    Tool(
        name="StockInfo",
        func=stock_info,
        description="Get stock information"
    )
]

agent = initialize_agent(
    tools,
    OpenAI(temperature=0),
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

print(agent.run("What's the current price of AAPL stock?"))