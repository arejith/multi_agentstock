import json

from langchain.tools import tool
from langchain_classic.agents import AgentType, initialize_agent

from Agents.analysis_team.helpers import parse_json_output, parse_request_payload
from Agents.supervisor.planner import SupervisorPlan

ANALYSIS_TEAM_PROMPT = (
    "You are the Analysis Team. You only read stored outputs from the Data Team and analyze them. "
    "Choose the minimum matching agent and return the tool result. "
    "Always read stored data first, then do analysis if needed."
)


class AnalysisTeam:
    system_prompt = ANALYSIS_TEAM_PROMPT
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION.value
    memory = None

    def __init__(self, analysis_tools: dict, llm):
        self.analysis_tools = analysis_tools
        self.llm = llm
        self.agent = initialize_agent(
            tools=self._build_team_tools(),
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            agent_kwargs={"prefix": ANALYSIS_TEAM_PROMPT},
        )

    def _build_team_tools(self):
        @tool
        def NewsAgent(request: str) -> str:
            """Read stored news first, then return news or sentiment analysis."""
            payload = parse_request_payload(request)
            action = payload.get("action", "read_news")
            if action == "analyze_sentiment":
                result = self.analysis_tools["get_sentiment_analysis"].invoke(
                    {"ticker": payload.get("ticker")}
                )
            else:
                result = self.analysis_tools["get_news"].invoke(
                    {
                        "query": payload.get("query"),
                        "ticker": payload.get("ticker"),
                    }
                )
            return json.dumps(result, default=str)

        @tool
        def FundamentalsAgent(request: str) -> str:
            """Read stored fundamentals first, then return fundamentals or risk analysis."""
            payload = parse_request_payload(request)
            action = payload.get("action", "read_fundamentals")
            if action == "analyze_risk":
                result = self.analysis_tools["get_risk_analysis"].invoke(
                    {"ticker": payload.get("ticker")}
                )
            else:
                result = self.analysis_tools["get_fundamentals"].invoke(
                    {"ticker": payload.get("ticker")}
                )
            return json.dumps(result, default=str)

        @tool
        def PredictionAgent(request: str) -> str:
            """Read stored transformer input first, then return input or transformer forecast."""
            payload = parse_request_payload(request)
            action = payload.get("action", "read_transformer_input")
            if action == "analyze_transformer":
                result = self.analysis_tools["run_transformer_analysis"].invoke(
                    {
                        "prepared_input_json": payload.get("prepared_input_json")
                    }
                )
            else:
                result = self.analysis_tools["get_transformer_input"].invoke(
                    {"ticker": payload.get("ticker")}
                )
            return json.dumps(result, default=str)

        return [
            NewsAgent,
            FundamentalsAgent,
            PredictionAgent,
        ]

    def execute(self, plan: SupervisorPlan, news_query: str) -> dict:
        outputs = {}
        if plan.needs_news or plan.needs_sentiment:
            result = self.analysis_tools["get_news"].invoke(
                {"query": news_query, "ticker": plan.ticker}
            )
            outputs["news"] = parse_json_output(result)
        if plan.needs_fundamentals or plan.needs_risk:
            result = self.analysis_tools["get_fundamentals"].invoke(
                {"ticker": plan.ticker}
            )
            outputs["fundamentals"] = parse_json_output(result)
        if plan.needs_prices or plan.needs_transformer:
            result = self.analysis_tools["get_transformer_input"].invoke(
                {"ticker": plan.ticker}
            )
            outputs["prepared_input"] = parse_json_output(result)
        if plan.needs_sentiment:
            result = self.analysis_tools["get_sentiment_analysis"].invoke(
                {"ticker": plan.ticker}
            )
            outputs["sentiment"] = parse_json_output(result)
        if plan.needs_risk:
            result = self.analysis_tools["get_risk_analysis"].invoke(
                {"ticker": plan.ticker}
            )
            outputs["risk"] = parse_json_output(result)
        if plan.needs_transformer:
            result = self.analysis_tools["run_transformer_analysis"].invoke(
                {
                    "prepared_input_json": json.dumps(
                        outputs.get("prepared_input", {}),
                        default=str,
                    )
                }
            )
            outputs["transformer"] = parse_json_output(result)
        return outputs
