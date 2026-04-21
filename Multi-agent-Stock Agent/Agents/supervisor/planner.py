from pydantic import BaseModel, Field

from Agents.supervisor.glossary import COMPANY_GLOSSARY, COMPANY_LOOKUP, glossary_text, normalize_text

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
        self.structured_llm = None
        if llm is not None and hasattr(llm, "with_structured_output"):
            try:
                self.structured_llm = llm.with_structured_output(SupervisorPlan)
            except Exception:
                self.structured_llm = None

    def create_plan(self, user_request: str) -> SupervisorPlan:
        plan = self._llm_plan(user_request)
        if plan and plan.ticker in COMPANY_LOOKUP:
            return complete_plan(plan)
        return fallback_plan(user_request)

    def _llm_plan(self, user_request: str) -> SupervisorPlan | None:
        if self.structured_llm is None:
            return None
        try:
            return self.structured_llm.invoke(
                SUPERVISOR_PROMPT.format(glossary=glossary_text()) + f"\nUser query: {user_request}"
            )
        except Exception:
            return None


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


def fallback_plan(user_request: str) -> SupervisorPlan:
    normalized = normalize_text(user_request)
    plan = None

    for ticker, company, sector, tool_sector in COMPANY_GLOSSARY:
        aliases = {
            normalize_text(ticker),
            normalize_text(company),
            normalize_text(company.replace("Inc.", "").replace("Corp.", "").replace("& Co.", "")),
        }
        if any(alias and alias in normalized for alias in aliases):
            plan = SupervisorPlan(
                ticker=ticker,
                company=company,
                sector=sector,
                tool_sector=tool_sector,
            )
            break

    if plan is None:
        return SupervisorPlan(action="invalid_request")

    if "fetch news" in normalized or normalized.startswith("news ") or " news " in f" {normalized} ":
        plan.action = "fetch_news"
        plan.needs_news = True
    elif "fetch fundamentals" in normalized or "fundamentals" in normalized:
        plan.action = "fetch_fundamentals"
        plan.needs_fundamentals = True
    elif "fetch transformer" in normalized:
        plan.action = "fetch_transformer_input"
        plan.needs_prices = bool(plan.tool_sector)
    elif "sentiment" in normalized:
        plan.action = "analyze_sentiment"
        plan.needs_news = True
        plan.needs_sentiment = True
    elif "risk" in normalized:
        plan.action = "analyze_risk"
        plan.needs_fundamentals = True
        plan.needs_risk = True
    elif "transformer" in normalized or "forecast" in normalized or "prediction" in normalized:
        plan.action = "analyze_transformer"
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


class PlannerAdapter:
    def __init__(self, resolver, router):
        self.resolver = resolver
        self.router = router

    def create_plan(self, user_request: str) -> SupervisorPlan:
        resolved = self.resolver.resolve(user_request)
        if not resolved.get("ticker"):
            return SupervisorPlan(action="invalid_request")

        action = "full_analysis"
        if self.router and hasattr(self.router, "route"):
            action = self.router.route(user_request).get("action", "full_analysis")

        entry = COMPANY_LOOKUP.get(resolved["ticker"], {})
        plan = SupervisorPlan(
            ticker=resolved["ticker"],
            company=resolved.get("company_name") or resolved.get("company") or entry.get("company"),
            sector=resolved.get("sector") or entry.get("sector"),
            tool_sector=entry.get("tool_sector"),
            action=action,
        )

        if plan.ticker in COMPANY_LOOKUP:
            return complete_plan(plan)

        if action == "fetch_news":
            plan.needs_news = True
        elif action == "fetch_fundamentals":
            plan.needs_fundamentals = True
        elif action == "fetch_transformer_input":
            plan.needs_prices = bool(plan.tool_sector)
        elif action == "analyze_sentiment":
            plan.needs_news = True
            plan.needs_sentiment = True
        elif action == "analyze_risk":
            plan.needs_fundamentals = True
            plan.needs_risk = True
        elif action == "analyze_transformer":
            plan.needs_prices = bool(plan.tool_sector)
            plan.needs_transformer = bool(plan.tool_sector)
        else:
            plan.needs_news = True
            plan.needs_fundamentals = True
            plan.needs_prices = bool(plan.tool_sector)
            plan.needs_sentiment = True
            plan.needs_risk = True
            plan.needs_transformer = bool(plan.tool_sector)
            plan.needs_decision = True

        plan.workflow = workflow_steps(plan)
        return plan
