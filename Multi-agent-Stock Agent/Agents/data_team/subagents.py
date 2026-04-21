import json
from dataclasses import dataclass

from langchain.tools import tool
from langchain_classic.agents import AgentType, initialize_agent

from Agents.data_team.helpers import parse_request_payload


@dataclass
class InitializedSubAgent:
    name: str
    agent_type: str
    function_name: str
    tool: object
    executor: object


def initialize_single_tool_subagent(llm, name: str, function_name: str, tool_obj, prompt: str):
    @tool(name)
    def wrapped_tool(request: str) -> str:
        """Single-input wrapper around one fixed stock function."""
        payload = parse_request_payload(request)
        result = tool_obj.invoke(payload)
        return json.dumps(result, default=str)

    executor = initialize_agent(
        tools=[wrapped_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        agent_kwargs={"prefix": prompt},
    )
    return InitializedSubAgent(
        name=name,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION.value,
        function_name=function_name,
        tool=wrapped_tool,
        executor=executor,
    )


def build_data_subagents(llm, data_tools: dict):
    return {
        "news": initialize_single_tool_subagent(
            llm,
            name="NewsFetchAgent",
            function_name="fetch_and_store_news",
            tool_obj=data_tools["fetch_and_store_news"],
            prompt="You are NewsFetchAgent. You only fetch and store news for the given ticker.",
        ),
        "fundamentals": initialize_single_tool_subagent(
            llm,
            name="FundamentalsFetchAgent",
            function_name="fetch_and_store_fundamentals",
            tool_obj=data_tools["fetch_and_store_fundamentals"],
            prompt="You are FundamentalsFetchAgent. You only fetch and store fundamentals for the given ticker.",
        ),
        "prices": initialize_single_tool_subagent(
            llm,
            name="StockDataFetchAgent",
            function_name="fetch_and_store_transformer_input",
            tool_obj=data_tools["fetch_and_store_transformer_input"],
            prompt="You are StockDataFetchAgent. You only build and store transformer-ready stock input for the given ticker.",
        ),
    }
