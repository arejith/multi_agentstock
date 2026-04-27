import json

from langchain.tools import tool
from langchain_classic.agents import AgentType, initialize_agent

from Agents.data_team.helpers import parse_json_output, parse_request_payload
from Agents.supervisor.planner import SupervisorPlan

DATA_TEAM_PROMPT = (
    "You are the Data Team. You only fetch and write data. Never analyze. "
    "Choose the minimum matching subagent and return the tool result. "
    "When using StockDataFetchAgent, you must pass both ticker and sector."
)


class DataTeam:
    system_prompt = DATA_TEAM_PROMPT
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION.value
    memory = None

    def __init__(self, data_tools: dict, llm):
        self.data_tools = data_tools
        self.llm = llm
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
            result = self.data_tools["fetch_and_store_news"].invoke(payload)
            return json.dumps(result, default=str)

        @tool
        def FundamentalsFetchAgent(request: str) -> str:
            """Fetch and store stock fundamentals only."""
            payload = parse_request_payload(request)
            result = self.data_tools["fetch_and_store_fundamentals"].invoke(payload)
            return json.dumps(result, default=str)

        @tool
        def StockDataFetchAgent(request: str) -> str:
            """Fetch and store transformer stock input only. Requires ticker and sector."""
            payload = parse_request_payload(request)
            result = self.data_tools["fetch_and_store_transformer_input"].invoke(payload)
            return json.dumps(result, default=str)

        return [NewsFetchAgent, FundamentalsFetchAgent, StockDataFetchAgent]

    def execute(self, plan: SupervisorPlan) -> dict:
        outputs = {}
        if plan.needs_news:
            result = self.data_tools["fetch_and_store_news"].invoke(
                {
                    "ticker": plan.ticker,
                    "sector": plan.tool_sector or plan.sector,
                    "company_name": plan.company,
                }
            )
            outputs["news"] = parse_json_output(result)
        if plan.needs_fundamentals:
            result = self.data_tools["fetch_and_store_fundamentals"].invoke(
                {"ticker": plan.ticker}
            )
            outputs["fundamentals"] = parse_json_output(result)
        if plan.needs_prices:
            result = self.data_tools["fetch_and_store_transformer_input"].invoke(
                {"ticker": plan.ticker, "sector": plan.tool_sector}
            )
            outputs["prices"] = parse_json_output(result)
        return outputs
