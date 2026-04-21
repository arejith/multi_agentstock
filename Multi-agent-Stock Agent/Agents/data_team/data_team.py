import json

from langchain.tools import tool
from langchain_classic.agents import AgentType, initialize_agent

from Agents.data_team.helpers import parse_json_output, parse_request_payload
from Agents.data_team.subagents import build_data_subagents
from Agents.supervisor.planner import SupervisorPlan

DATA_TEAM_PROMPT = (
    "You are the Data Team. You only fetch and write data. Never analyze. "
    "Choose the minimum matching subagent and return the tool result."
)


class DataTeam:
    system_prompt = DATA_TEAM_PROMPT
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION.value
    memory = None

    def __init__(self, data_tools: dict, llm):
        self.data_tools = data_tools
        self.llm = llm
        self.subagents = build_data_subagents(llm, data_tools)
        self.agent = initialize_agent(
            tools=self._build_team_tools(),
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            agent_kwargs={"prefix": DATA_TEAM_PROMPT},
        )

    def _build_team_tools(self):
        @tool
        def NewsFetchAgent(request: str) -> str:
            """Fetch and store latest stock news only."""
            payload = parse_request_payload(request)
            payload = self.subagents["news"].executor.run(
                "Use NewsFetchAgent with this exact JSON request and return only the tool output: "
                f"{json.dumps(payload, default=str)}"
            )
            return payload if isinstance(payload, str) else json.dumps(payload)

        @tool
        def FundamentalsFetchAgent(request: str) -> str:
            """Fetch and store stock fundamentals only."""
            payload = parse_request_payload(request)
            payload = self.subagents["fundamentals"].executor.run(
                "Use FundamentalsFetchAgent with this exact JSON request and return only the tool output: "
                f"{json.dumps(payload, default=str)}"
            )
            return payload if isinstance(payload, str) else json.dumps(payload)

        @tool
        def StockDataFetchAgent(request: str) -> str:
            """Fetch and store transformer stock input only."""
            payload = parse_request_payload(request)
            payload = self.subagents["prices"].executor.run(
                "Use StockDataFetchAgent with this exact JSON request and return only the tool output: "
                f"{json.dumps(payload, default=str)}"
            )
            return payload if isinstance(payload, str) else json.dumps(payload)

        return [NewsFetchAgent, FundamentalsFetchAgent, StockDataFetchAgent]

    def execute(self, plan: SupervisorPlan) -> dict:
        outputs = {}
        if plan.needs_news:
            payload = self.subagents["news"].tool.invoke(
                json.dumps(
                    {"ticker": plan.ticker, "sector": plan.tool_sector or plan.sector, "company_name": plan.company},
                    default=str,
                )
            )
            outputs["news"] = parse_json_output(payload)
        if plan.needs_fundamentals:
            payload = self.subagents["fundamentals"].tool.invoke(json.dumps({"ticker": plan.ticker}, default=str))
            outputs["fundamentals"] = parse_json_output(payload)
        if plan.needs_prices:
            payload = self.subagents["prices"].tool.invoke(
                json.dumps({"ticker": plan.ticker, "sector": plan.tool_sector}, default=str)
            )
            outputs["prices"] = parse_json_output(payload)
        return outputs
