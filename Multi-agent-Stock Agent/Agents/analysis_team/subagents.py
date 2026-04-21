import json
from dataclasses import dataclass

from langchain.tools import tool
from langchain_classic.agents import AgentType, initialize_agent

from Agents.analysis_team.helpers import parse_request_payload


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


def build_analysis_subagents(llm, analysis_tools: dict):
    return {
        "news_read": initialize_single_tool_subagent(
            llm,
            name="NewsReadAgent",
            function_name="get_news",
            tool_obj=analysis_tools["get_news"],
            prompt="You are NewsReadAgent. You only read stored news for the given ticker and query.",
        ),
        "sentiment": initialize_single_tool_subagent(
            llm,
            name="SentimentAgent",
            function_name="get_sentiment_analysis",
            tool_obj=analysis_tools["get_sentiment_analysis"],
            prompt="You are SentimentAgent. You only perform sentiment analysis on stored news.",
        ),
        "fundamentals_read": initialize_single_tool_subagent(
            llm,
            name="FundamentalsReadAgent",
            function_name="get_fundamentals",
            tool_obj=analysis_tools["get_fundamentals"],
            prompt="You are FundamentalsReadAgent. You only read stored fundamentals.",
        ),
        "risk": initialize_single_tool_subagent(
            llm,
            name="RiskAgent",
            function_name="get_risk_analysis",
            tool_obj=analysis_tools["get_risk_analysis"],
            prompt="You are RiskAgent. You only analyze risk from stored fundamentals.",
        ),
        "transformer_read": initialize_single_tool_subagent(
            llm,
            name="TransformerReadAgent",
            function_name="get_transformer_input",
            tool_obj=analysis_tools["get_transformer_input"],
            prompt="You are TransformerReadAgent. You only read stored transformer input.",
        ),
        "transformer": initialize_single_tool_subagent(
            llm,
            name="TransformerAgent",
            function_name="run_transformer_analysis",
            tool_obj=analysis_tools["run_transformer_analysis"],
            prompt="You are TransformerAgent. You only run the transformer model on prepared input.",
        ),
    }
