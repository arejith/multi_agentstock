import json

from langchain.tools import tool
from langchain_classic.agents import AgentType, initialize_agent

from Agents.analysis_team.helpers import parse_json_output, parse_request_payload
from Agents.analysis_team.subagents import build_analysis_subagents
from Agents.supervisor.planner import SupervisorPlan

ANALYSIS_TEAM_PROMPT = (
    "You are the Analysis Team. You only read stored outputs from the Data Team and analyze them. "
    "Choose the minimum matching subagent and return the tool result."
)


class AnalysisTeam:
    system_prompt = ANALYSIS_TEAM_PROMPT
    agent_type = AgentType.ZERO_SHOT_REACT_DESCRIPTION.value
    memory = None

    def __init__(self, analysis_tools: dict, llm):
        self.analysis_tools = analysis_tools
        self.llm = llm
        self.subagents = build_analysis_subagents(llm, analysis_tools)
        self.agent = initialize_agent(
            tools=self._build_team_tools(),
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=False,
            agent_kwargs={"prefix": ANALYSIS_TEAM_PROMPT},
        )

    def _build_team_tools(self):
        @tool
        def NewsReadAgent(request: str) -> str:
            """Read stored news only."""
            payload = parse_request_payload(request)
            payload = self.subagents["news_read"].executor.run(
                "Use NewsReadAgent with this exact JSON request and return only the tool output: "
                f"{json.dumps(payload, default=str)}"
            )
            return payload if isinstance(payload, str) else json.dumps(payload)

        @tool
        def SentimentAgent(request: str) -> str:
            """Run sentiment analysis only."""
            payload = parse_request_payload(request)
            payload = self.subagents["sentiment"].executor.run(
                "Use SentimentAgent with this exact JSON request and return only the tool output: "
                f"{json.dumps(payload, default=str)}"
            )
            return payload if isinstance(payload, str) else json.dumps(payload)

        @tool
        def FundamentalsReadAgent(request: str) -> str:
            """Read stored fundamentals only."""
            payload = parse_request_payload(request)
            payload = self.subagents["fundamentals_read"].executor.run(
                "Use FundamentalsReadAgent with this exact JSON request and return only the tool output: "
                f"{json.dumps(payload, default=str)}"
            )
            return payload if isinstance(payload, str) else json.dumps(payload)

        @tool
        def RiskAgent(request: str) -> str:
            """Run risk analysis only."""
            payload = parse_request_payload(request)
            payload = self.subagents["risk"].executor.run(
                "Use RiskAgent with this exact JSON request and return only the tool output: "
                f"{json.dumps(payload, default=str)}"
            )
            return payload if isinstance(payload, str) else json.dumps(payload)

        @tool
        def TransformerReadAgent(request: str) -> str:
            """Read stored transformer input only."""
            payload = parse_request_payload(request)
            payload = self.subagents["transformer_read"].executor.run(
                "Use TransformerReadAgent with this exact JSON request and return only the tool output: "
                f"{json.dumps(payload, default=str)}"
            )
            return payload if isinstance(payload, str) else json.dumps(payload)

        @tool
        def TransformerAgent(request: str) -> str:
            """Run transformer analysis only."""
            payload = parse_request_payload(request)
            payload = self.subagents["transformer"].executor.run(
                "Use TransformerAgent with this exact JSON request and return only the tool output: "
                f"{json.dumps(payload, default=str)}"
            )
            return payload if isinstance(payload, str) else json.dumps(payload)

        return [
            NewsReadAgent,
            SentimentAgent,
            FundamentalsReadAgent,
            RiskAgent,
            TransformerReadAgent,
            TransformerAgent,
        ]

    def execute(self, plan: SupervisorPlan, news_query: str) -> dict:
        outputs = {}
        if plan.needs_news or plan.needs_sentiment:
            payload = self.subagents["news_read"].tool.invoke(
                json.dumps({"query": news_query, "ticker": plan.ticker}, default=str)
            )
            outputs["news"] = parse_json_output(payload)
        if plan.needs_fundamentals or plan.needs_risk:
            payload = self.subagents["fundamentals_read"].tool.invoke(json.dumps({"ticker": plan.ticker}, default=str))
            outputs["fundamentals"] = parse_json_output(payload)
        if plan.needs_prices or plan.needs_transformer:
            payload = self.subagents["transformer_read"].tool.invoke(json.dumps({"ticker": plan.ticker}, default=str))
            outputs["prepared_input"] = parse_json_output(payload)
        if plan.needs_sentiment:
            payload = self.subagents["sentiment"].tool.invoke(json.dumps({"ticker": plan.ticker}, default=str))
            outputs["sentiment"] = parse_json_output(payload)
        if plan.needs_risk:
            payload = self.subagents["risk"].tool.invoke(json.dumps({"ticker": plan.ticker}, default=str))
            outputs["risk"] = parse_json_output(payload)
        if plan.needs_transformer:
            prepared = outputs.get("prepared_input", {})
            if not prepared or prepared.get("found") is False:
                outputs["transformer"] = {"ticker": plan.ticker, "prediction": None, "signal": "UNKNOWN"}
            else:
                payload = self.subagents["transformer"].tool.invoke(
                    json.dumps({"prepared_input_json": json.dumps(prepared, default=str)}, default=str)
                )
                outputs["transformer"] = parse_json_output(payload)
        return outputs
