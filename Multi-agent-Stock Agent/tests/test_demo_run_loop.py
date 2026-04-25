import io
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import Agents.supervisor.runtime as runtime_module
import demo_run_loop

from tests.test_agent_pipeline import build_test_runtime


def capture_output(function, *args, **kwargs):
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        function(*args, **kwargs)
    return buffer.getvalue()


def install_demo_test_runtime():
    runtime_module._RUNTIME = build_test_runtime()


def run_demo_loop_test():
    install_demo_test_runtime()

    examples_output = capture_output(demo_run_loop.print_examples)
    if "Demo Prompts" not in examples_output:
        raise AssertionError("Demo examples should list regular prompts")
    if "Edge Case Prompts" not in examples_output:
        raise AssertionError("Demo examples should list edge case prompts")
    if len(demo_run_loop.DEMO_REQUESTS) < 4:
        raise AssertionError("Demo showcase should include multiple normal prompts")
    if len(demo_run_loop.EDGE_CASE_REQUESTS) < 8:
        raise AssertionError("Demo edge cases should cover invalid, fetch-only, and analysis-only paths")

    synthetic_result = {
        "request_resolution": {"ticker": "AAPL", "company_name": "Apple Inc.", "sector": "Information Technology"},
        "flow": {"workflow": ["News_Agent(Data Team)", "Decision_Agent(Decision Layer)"]},
        "final_report": {"decision": "BUY", "reasoning": ["LLM weighed the forecast and risk."], "risks": []},
    }
    formatted = demo_run_loop.summarize_result(synthetic_result)
    if "# Why This Decision" not in formatted:
        raise AssertionError("Demo summary should explain why the decision was made")
    if "LLM weighed the forecast and risk." not in formatted:
        raise AssertionError("Demo summary should include decision reasoning")

    showcase_output = capture_output(demo_run_loop.run_showcase, show_raw=False)
    if showcase_output.count("Request:") != len(demo_run_loop.DEMO_REQUESTS):
        raise AssertionError("Showcase should run every normal demo prompt")
    if "# Why This Decision" not in showcase_output:
        raise AssertionError("Showcase output should include why decisions were made")
    if "error_type" in showcase_output:
        raise AssertionError(f"Showcase should not raise exceptions:\n{showcase_output}")

    edge_output = capture_output(demo_run_loop.run_edge_cases, show_raw=False)
    if edge_output.count("Request:") != len(demo_run_loop.EDGE_CASE_REQUESTS):
        raise AssertionError("Edge-case run should execute every edge prompt")
    if "Unsupported stock query." not in edge_output:
        raise AssertionError("Edge cases should include invalid request handling")
    if "The predicted daily return for next day is" not in edge_output:
        raise AssertionError("Edge cases should exercise transformer forecast output")
    if "error_type" in edge_output:
        raise AssertionError(f"Edge cases should not raise exceptions:\n{edge_output}")

    raw_output = capture_output(demo_run_loop.run_request, "Just fetch news for Apple", show_raw=True)
    if "Raw Output" not in raw_output:
        raise AssertionError("Raw mode should print the raw JSON payload")
    raw_json = raw_output.split("=== Raw Output ===", 1)[1].strip()
    parsed = json.loads(raw_json)
    if parsed["request_route"]["action"] != "fetch_news":
        raise AssertionError(f"Raw fetch-news output should preserve routing: {parsed['request_route']}")

    print("Demo loop edge cases executed successfully.")


if __name__ == "__main__":
    run_demo_loop_test()
