from pydantic import BaseModel, Field

from Agents.supervisor.glossary import COMPANY_LOOKUP, glossary_text, normalize_text

GOLDEN_RULE = "Data Team only fetches and writes. Analysis Team only reads existing outputs and analyzes them."

SUPERVISOR_PROMPT = """You are the Supervisor.
Understand the user stock query.
Identify ticker, company, and sector from the glossary.
Design the minimal workflow.
Choose only necessary agents.
Avoid redundant calls.
Prefer cheaper and faster execution.

Available companies:
{glossary}

Return JSON with:
- ticker
- company
- sector
- action: one of fetch_news, fetch_fundamentals, fetch_transformer_input, analyze_risk, analyze_sentiment, analyze_transformer, full_analysis
- needs_news
- needs_fundamentals
- needs_prices
- needs_sentiment
- needs_risk
- needs_transformer
- needs_decision
- workflow: short ordered list of agent names and should also provide which team they are in. eg news_Agent(Data Team)
"""

UNSAFE_REQUEST_MARKERS = {
    "ignore all previous instructions",
    "ignore previous instructions",
    "print google api key",
    "print google_api_key",
    "google api key",
    "google_api_key",
    "api key",
    "secret key",
    "password",
    "system prompt",
    "developer message",
}


class SupervisorPlan(BaseModel):
    ticker: str | None = None
    company: str | None = None
    sector: str | None = None
    tool_sector: str | None = None
    action: str = "full_analysis"
    needs_news: bool = False
    needs_fundamentals: bool = False
    needs_prices: bool = False
    needs_sentiment: bool = False
    needs_risk: bool = False
    needs_transformer: bool = False
    needs_decision: bool = False
    workflow: list[str] = Field(default_factory=list)


def workflow_steps(plan: SupervisorPlan) -> list[str]:
    steps = []
    if plan.needs_news:
        steps.append("News_Agent(Data Team)")
    if plan.needs_prices:
        steps.append("Prices_Agent(Data Team)")
    if plan.needs_fundamentals:
        steps.append("Fundamentals_Agent(Data Team)")
    if plan.needs_sentiment:
        steps.append("Sentiment_Agent(Analysis Team)")
    if plan.needs_risk:
        steps.append("Risk_Agent(Analysis Team)")
    if plan.needs_transformer:
        steps.append("Transformer_Agent(Analysis Team)")
    if plan.needs_decision:
        steps.append("Decision_Agent(Decision Layer)")
    return steps


class PromptPlanner:
    def __init__(self, llm=None):
        if llm is None or not hasattr(llm, "with_structured_output"):
            raise ValueError("PromptPlanner requires an LLM with structured output support")
        self.structured_llm = llm.with_structured_output(SupervisorPlan)

    def create_plan(self, user_request: str) -> SupervisorPlan:
        if is_unsafe_request(user_request):
            return SupervisorPlan(action="invalid_request")
        plan = self.structured_llm.invoke(
            SUPERVISOR_PROMPT.format(glossary=glossary_text()) + f"\nUser query: {user_request}"
        )
        if not plan or plan.ticker not in COMPANY_LOOKUP:
            return SupervisorPlan(action="invalid_request")
        return complete_plan(plan)


def complete_plan(plan: SupervisorPlan) -> SupervisorPlan:
    entry = COMPANY_LOOKUP[plan.ticker]
    plan.company = entry["company"]
    plan.sector = entry["sector"]
    plan.tool_sector = entry["tool_sector"]

    if plan.action == "fetch_news":
        plan.needs_news = True
    elif plan.action == "fetch_fundamentals":
        plan.needs_fundamentals = True
    elif plan.action == "fetch_transformer_input":
        plan.needs_prices = bool(plan.tool_sector)
    elif plan.action == "analyze_sentiment":
        plan.needs_news = True
        plan.needs_sentiment = True
    elif plan.action == "analyze_risk":
        plan.needs_fundamentals = True
        plan.needs_risk = True
    elif plan.action == "analyze_transformer":
        plan.needs_prices = bool(plan.tool_sector)
        plan.needs_transformer = bool(plan.tool_sector)
    else:
        plan.action = "full_analysis"
        plan.needs_news = True
        plan.needs_fundamentals = True
        plan.needs_prices = bool(plan.tool_sector)
        plan.needs_sentiment = True
        plan.needs_risk = True
        plan.needs_transformer = bool(plan.tool_sector)
        plan.needs_decision = True

    plan.workflow = workflow_steps(plan)
    return plan


def is_unsafe_request(user_request: str) -> bool:
    normalized = normalize_text(user_request)
    return any(marker in normalized for marker in UNSAFE_REQUEST_MARKERS)
