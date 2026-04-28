from dataclasses import dataclass

from config import get_default_llm
from memory.retrievers.fundamentals_retriever import FundamentalsRetriever
from memory.retrievers.news_retriever import NewsRetriever
from memory.retrievers.transformer_retriever import TransformerInputRetriever
from memory.stores.fundamentals_store import FundamentalsStore
from memory.stores.news_store import NewsVectorStore
from memory.stores.transformer_store import TransformerInputStore
from memory.writers.data_writer import DataWriter
from Agents.analysis_team.decision import DecisionLayer
from Agents.analysis_team.analysis_team import AnalysisTeam
from Agents.data_team.data_team import DataTeam
from Agents.supervisor.glossary import COMPANY_LOOKUP
from Agents.supervisor.planner import GOLDEN_RULE, PromptPlanner, SupervisorPlan, workflow_steps
from tools.tools_registry import get_tool_sets


@dataclass
class RuntimeStores:
    news_store: NewsVectorStore
    fundamentals_store: FundamentalsStore
    transformer_store: TransformerInputStore


@dataclass
class RuntimeRetrievers:
    news_retriever: NewsRetriever
    fundamentals_retriever: FundamentalsRetriever
    transformer_retriever: TransformerInputRetriever


@dataclass
class RuntimeTools:
    data_tools: dict
    analysis_tools: dict


@dataclass
class RuntimeServices:
    supervisor: "SupervisorAgent"
    decision_layer: DecisionLayer


@dataclass
class PipelineRuntime:
    stores: RuntimeStores
    writer: DataWriter
    retrievers: RuntimeRetrievers
    tools: RuntimeTools
    services: RuntimeServices


class SupervisorAgent:
   

    def __init__(self, planner, data_team: DataTeam, analysis_team: AnalysisTeam, decision_layer: DecisionLayer, llm):
        self.planner = planner
        self.data_team = data_team
        self.analysis_team = analysis_team
        self.decision_layer = decision_layer
        self.memory = None
        self.llm = llm

    def run(self, ticker: str, sector: str, news_query: str | None = None, company_name: str | None = None):
        plan = self._build_full_analysis_plan(ticker, sector, company_name=company_name)
        return self._execute(plan, user_request=None, news_query=news_query or ticker)

    def run_request(self, user_request: str):
        plan = self.planner.create_plan(user_request)
        if plan.action == "invalid_request" or not plan.ticker:
            return invalid_result(user_request)
        return self._execute(plan, user_request=user_request, news_query=plan.company or plan.ticker)

    def _execute(self, plan: SupervisorPlan, user_request: str | None, news_query: str):
        self._refresh_workflow(plan)
        raw_data_phase = self.data_team.execute(plan)
        raw_analysis_phase = self.analysis_team.execute(plan, news_query)
        data_phase = build_data_phase(plan, raw_data_phase, raw_analysis_phase)
        analysis_phase = build_analysis_phase(plan, news_query, raw_analysis_phase)
        final_report = build_final_report(plan, analysis_phase, self.decision_layer)
        result = {
            "request": {
                "ticker": plan.ticker,
                "company": plan.company,
                "sector": plan.sector,
                "news_query": news_query,
                "user_request": user_request,
            },
            "request_resolution": {
                "ticker": plan.ticker,
                "company_name": plan.company,
                "sector": plan.sector,
            },
            "request_route": {
                "action": plan.action,
                "workflow": plan.workflow,
            },
            "flow": {
                "style": "simple_prompt_driven_multi_agent",
                "golden_rule": GOLDEN_RULE,
                "workflow": plan.workflow,
            },
            "data_phase": data_phase,
            "analysis_phase": analysis_phase,
            "final_report": final_report,
        }
        result["user_output"] = build_user_output(result)
        return result

    def _build_full_analysis_plan(self, ticker: str, sector: str, company_name: str | None = None):
        entry = COMPANY_LOOKUP.get(
            ticker,
            {"company": company_name or ticker, "sector": sector, "tool_sector": sector},
        )
        plan = SupervisorPlan(
            ticker=ticker,
            company=company_name or entry["company"],
            sector=entry["sector"],
            tool_sector=entry.get("tool_sector") or sector,
            action="full_analysis",
            needs_news=True,
            needs_fundamentals=True,
            needs_prices=bool(entry.get("tool_sector") or sector),
            needs_sentiment=True,
            needs_risk=True,
            needs_transformer=bool(entry.get("tool_sector") or sector),
            needs_decision=True,
        )
        self._refresh_workflow(plan)
        return plan

    def _refresh_workflow(self, plan: SupervisorPlan):
        plan.workflow = workflow_steps(plan)


def build_data_phase(plan: SupervisorPlan, raw_data_phase: dict, raw_analysis_phase: dict):
    news_payload = raw_data_phase.get("news")
    fundamentals_payload = raw_data_phase.get("fundamentals")
    transformer_payload = raw_data_phase.get("prices")

    if plan.action == "fetch_news":
        news_payload = {"ticker": plan.ticker, "news": raw_analysis_phase.get("news", []), "status": "success"}
    if plan.action == "fetch_fundamentals":
        fundamentals = raw_analysis_phase.get("fundamentals")
        fundamentals_payload = {
            "ticker": plan.ticker,
            "fundamentals": [] if not fundamentals or fundamentals.get("found") is False else fundamentals,
            "status": "success",
        }
    if plan.action == "fetch_transformer_input":
        prepared = raw_analysis_phase.get("prepared_input")
        transformer_payload = {
            "ticker": plan.ticker,
            "transformer_input": [] if not prepared or prepared.get("found") is False else prepared,
            "status": "success",
        }

    return {
        "news": news_payload,
        "fundamentals": fundamentals_payload,
        "transformer_input": transformer_payload,
    }


def build_analysis_phase(plan: SupervisorPlan, news_query: str, raw_analysis_phase: dict):
    return {
        "ticker": plan.ticker,
        "news_analysis": {
            "query": news_query,
            "news": raw_analysis_phase.get("news", []),
            "sentiment": raw_analysis_phase.get("sentiment"),
        },
        "risk_analysis": {
            "fundamentals": raw_analysis_phase.get("fundamentals"),
            "risk": raw_analysis_phase.get("risk"),
        },
        "transformer_analysis": {
            "prepared_input": raw_analysis_phase.get("prepared_input"),
            "transformer": raw_analysis_phase.get("transformer"),
        },
    }


def build_final_report(plan: SupervisorPlan, analysis_phase: dict, decision_layer: DecisionLayer):
    quick_report = build_quick_report(plan, analysis_phase)
    if quick_report is not None:
        return quick_report
    return decision_layer.decide(
        ticker=plan.ticker,
        company=plan.company,
        sector=plan.sector,
        analysis_outputs=analysis_phase,
    )


def build_quick_report(plan: SupervisorPlan, analysis_phase: dict):
    if plan.action == "fetch_news":
        news_count = len(analysis_phase.get("news_analysis", {}).get("news") or [])
        return {
            "recommendation": "data_fetched" if news_count else "no_data_found",
            "decision": None,
            "confidence": None,
            "reasoning": [f"Fetched {news_count} news item(s) for {plan.ticker}."],
            "risks": [],
        }
    if plan.action == "fetch_fundamentals":
        fundamentals = analysis_phase.get("risk_analysis", {}).get("fundamentals")
        return {
            "recommendation": "data_fetched" if fundamentals else "no_data_found",
            "decision": None,
            "confidence": None,
            "reasoning": [f"Fetched fundamentals for {plan.ticker}."],
            "risks": [],
        }
    if plan.action == "fetch_transformer_input":
        prepared = analysis_phase.get("transformer_analysis", {}).get("prepared_input")
        return {
            "recommendation": "data_fetched" if prepared else "no_data_found",
            "decision": None,
            "confidence": None,
            "reasoning": [f"Built transformer input for {plan.ticker}."],
            "risks": [],
        }
    if plan.action == "analyze_sentiment":
        sentiment = analysis_phase.get("news_analysis", {}).get("sentiment") or {}
        return {
            "recommendation": "analysis_complete",
            "decision": None,
            "confidence": None,
            "reasoning": [f"Sentiment score: {sentiment.get('score')}"],
            "risks": [],
            "sentiment_score": sentiment.get("score"),
        }
    if plan.action == "analyze_risk":
        risk = analysis_phase.get("risk_analysis", {}).get("risk") or {}
        return {
            "recommendation": "analysis_complete",
            "decision": None,
            "confidence": None,
            "reasoning": [f"Risk level: {risk.get('risk_level')}"],
            "risks": risk.get("reasons") or [],
            "risk_level": risk.get("risk_level"),
        }
    if plan.action == "analyze_transformer":
        transformer = analysis_phase.get("transformer_analysis", {}).get("transformer") or {}
        return {
            "recommendation": "analysis_complete",
            "decision": None,
            "confidence": None,
            "reasoning": [f"Transformer forecast: {transformer.get('signal')}"],
            "risks": [],
            "transformer_signal": transformer.get("signal"),
            "prediction": transformer.get("prediction"),
        }
    return None


def build_user_output(result: dict):
    request = result["request"]
    final_report = result["final_report"]
    workflow = " -> ".join(result["flow"]["workflow"]) or "N/A"
    decision = final_report.get("decision") or final_report.get("recommendation", "").upper()
    confidence = final_report.get("confidence")
    if isinstance(confidence, (int, float)):
        confidence_text = f"{round(confidence * 100)}%" if confidence <= 1 else f"{confidence}%"
    else:
        confidence_text = "N/A"

    lines = [
        "# Company Identified",
        f"{request['ticker']} | {request['company']} | {request['sector']}",
        "",
        "# Workflow Used",
        workflow,
        "",
        "# Why This Decision",
    ]
    for item in final_report.get("reasoning") or []:
        lines.append(f"- {item}")
    lines.extend(["", "# Decision", str(decision), "", "# Confidence", confidence_text, "", "# Risks"])
    for risk in final_report.get("risks") or ["No major risk flag from current outputs."]:
        lines.append(f"- {risk}")

    return {
        "status": "success",
        "action": result["request_route"]["action"],
        "ticker": request["ticker"],
        "response": "\n".join(lines),
    }


def invalid_result(user_request: str):
    result = {
        "request": {"ticker": None, "company": None, "sector": None, "user_request": user_request},
        "request_resolution": {"ticker": None, "company_name": None, "sector": None},
        "request_route": {"action": "invalid_request", "workflow": []},
        "flow": {"style": "simple_prompt_driven_multi_agent", "golden_rule": GOLDEN_RULE, "workflow": []},
        "data_phase": {},
        "analysis_phase": {},
        "final_report": {
            "recommendation": "invalid_request",
            "decision": None,
            "confidence": None,
            "reasoning": ["Could not identify a supported stock from the query."],
            "risks": [],
            "status": "error",
        },
    }
    result["user_output"] = {
        "status": "error",
        "action": "invalid_request",
        "ticker": None,
        "response": "Unsupported stock query.",
    }
    return result


def build_runtime(*, supervisor_llm=None, data_llm=None, analysis_llm=None, request_resolver=None, request_router=None):
    shared_llm = supervisor_llm or get_default_llm()

    news_store = NewsVectorStore()
    fundamentals_store = FundamentalsStore()
    transformer_store = TransformerInputStore()
    writer = DataWriter(news_store, fundamentals_store, transformer_store)

    news_retriever = NewsRetriever(news_store)
    fundamentals_retriever = FundamentalsRetriever(fundamentals_store)
    transformer_retriever = TransformerInputRetriever(transformer_store)

    tool_sets = get_tool_sets(
        news_retriever,
        fundamentals_retriever,
        transformer_retriever,
        writer,
        keyword_llm=shared_llm,
    )
    data_tools = {tool.name: tool for tool in tool_sets["data_tools"]}
    analysis_tools = {tool.name: tool for tool in tool_sets["analysis_tools"]}

    planner = PromptPlanner(shared_llm)
    decision_layer = DecisionLayer(llm=analysis_llm or shared_llm)
    supervisor = SupervisorAgent(
        planner=planner,
        data_team=DataTeam(data_tools, data_llm or shared_llm),
        analysis_team=AnalysisTeam(analysis_tools, analysis_llm or shared_llm),
        decision_layer=decision_layer,
        llm=shared_llm,
    )

    return PipelineRuntime(
        stores=RuntimeStores(news_store, fundamentals_store, transformer_store),
        writer=writer,
        retrievers=RuntimeRetrievers(news_retriever, fundamentals_retriever, transformer_retriever),
        tools=RuntimeTools(data_tools, analysis_tools),
        services=RuntimeServices(supervisor=supervisor, decision_layer=decision_layer),
    )


_RUNTIME = None


def get_runtime() -> PipelineRuntime:
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = build_runtime()
    return _RUNTIME


def reset_runtime():
    global _RUNTIME
    _RUNTIME = None


def run_stock_analysis(
    ticker: str | None = None,
    sector: str | None = None,
    news_query: str | None = None,
    user_request: str | None = None,
):
    runtime = get_runtime()
    if user_request:
        return runtime.services.supervisor.run_request(user_request)
    if ticker is None or sector is None:
        raise ValueError("ticker and sector are required unless user_request is provided")
    return runtime.services.supervisor.run(ticker=ticker, sector=sector, news_query=news_query)
