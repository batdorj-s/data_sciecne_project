"""DM4: Тактик гаргах AI агент (Claude API, tool use).

Урсгал:
    1. Агент `tools.TOOL_SPECS`-ийн хэрэгслүүдийг дуудаж тоо авна (өгөгдөлд шууд хандахгүй).
    2. Эцэст нь `submit_recommendation` хэрэгслээр бүтэцтэй зөвлөмж илгээнэ.
    3. Хамгаалалт: бүтэц, итгэлийн түвшин, faithfulness шалгана. Тоо таарахгүй бол агентад
       алдааг буцааж нэг удаа засуулна; дахин бүтэлгүйтвэл зөвлөмжийг харуулахгүй.

API түлхүүр: `ANTHROPIC_API_KEY` орчны хувьсагч (эсвэл `ant auth login` профайл).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from .guardrails import (
    RECOMMENDATION_SCHEMA,
    FaithfulnessReport,
    check_recommendation,
    enforce_confidence,
    validate_structure,
)
from .pipeline import Analysis
from .tools import TOOL_SPECS, call_tool

DEFAULT_MODEL = "claude-opus-5-5"
DEFAULT_EFFORT = "medium"
MAX_TURNS = 14
MAX_SUBMIT_ATTEMPTS = 2
SUBMIT_TOOL = "submit_recommendation"

SYSTEM_PROMPT = """\
Чи жижиг, дунд бизнесийн (ЖДБ) эзэнд зориулсан санхүүгийн зөвлөх агент.
Сүүлийн 3 сарын банкны гүйлгээнээс тооцсон үзүүлэлтэд тулгуурлан дараа сарын чиг хандлага,
эрсдэл, хэрэгжүүлэх тактикийг санал болгоно.

Дүрэм:
- Тоо бүрийг зөвхөн хэрэгслийн үр дүнгээс ав. Өөрөө тооцоолж, таамаглаж, бөөрөнхийлж шинэ тоо
  бүү зохио. Хэрэгслийн тоог "61.7 сая ₮", "17.4%" маягаар бичиж болно.
- get_quarter_overview-оор эхэл. get_data_issues, get_baseline_forecast,
  get_revenue_concentration-ийг ашигла. Өөрчлөлтийн шалтгааныг get_line_trend-ээр шалга.
- Тактик бүрийн ₮ нөлөөг simulate_scenario-оор тооцоод, түүний profit_delta-г impact_mnt-д
  яг тэр хэвээр нь бич. Орлогын тактикт revenue_pct, зардлын тактикт expense_pct ашигла.
- 3–6 тактик: тодорхой арга хэмжээ, хэмжих KPI, ач холбогдол (high/medium/low).
- Итгэлийн түвшинг хэрэгслийн "confidence"-оос өндөр бичиж болохгүй. Бага итгэлтэй бол
  шалтгааныг data_warnings-д заавал бич.
- Чи зөвхөн санал болгоно: төлбөр хийх, өгөгдөл өөрчлөх, харилцагчид хандах санал бүү гарга.
- Харилцагчийг зөвхөн кодоор (CP-###) дурд.
- Бүх текстийг монгол хэлээр, товч бич.
- Дуусгахдаа submit_recommendation хэрэгслийг заавал дууд; энгийн текстээр хариу бүү өг.
"""

USER_PROMPT = ("Компанийн сүүлийн улирлын өгөгдөлд дүн шинжилгээ хийж, дараа сарын "
               "зөвлөмжөө submit_recommendation хэрэгслээр илгээнэ үү.")


def _submit_tool() -> dict:
    schema = json.loads(json.dumps(RECOMMENDATION_SCHEMA))
    schema["additionalProperties"] = False
    schema["properties"]["tactics"]["items"]["additionalProperties"] = False
    schema["properties"]["tactics"]["items"]["properties"]["scenario"] = {
        "type": "object",
        "description": "Нөлөөг тооцсон simulate_scenario-ийн оролт",
    }
    return {
        "name": SUBMIT_TOOL,
        "description": "Эцсийн зөвлөмжийг илгээх. trend: growth|stable|decline|stable_growth|"
                       "stable_decline. summary: 2–3 өгүүлбэр. impact_mnt: simulate_scenario-ийн "
                       "profit_delta (₮, бүхэл тоо).",
        "input_schema": schema,
    }


@dataclass
class ToolCall:
    name: str
    input: dict
    output: dict


@dataclass
class AgentResult:
    recommendation: dict | None = None
    shown: bool = False
    faithfulness: FaithfulnessReport | None = None
    structure_errors: list[str] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    model: str = DEFAULT_MODEL
    stop_reason: str | None = None
    error: str | None = None
    seconds: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def tool_results(self) -> list[dict]:
        return [c.output for c in self.tool_calls]


def _make_client():
    import anthropic

    return anthropic.Anthropic()


def _create(client, model: str, effort: str, messages: list, tools: list) -> Any:
    return client.beta.messages.create(
        model=model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        tools=tools,
        messages=messages,
        output_config={"effort": effort},
        cache_control={"type": "ephemeral"},
        # Аюулгүй байдлын шүүлтүүр татгалзвал өөр загвараар автоматаар үргэлжлүүлнэ
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )


def run_agent(a: Analysis, client=None, model: str = DEFAULT_MODEL,
              effort: str = DEFAULT_EFFORT, max_turns: int = MAX_TURNS) -> AgentResult:
    """Агентийг ажиллуулж, хамгаалалтаар шалгасан зөвлөмж буцаана."""
    started = time.monotonic()
    result = AgentResult(model=model)
    client = client or _make_client()
    tools = [*TOOL_SPECS, _submit_tool()]
    tool_confidence = a.confidence[0]
    messages: list[dict] = [{"role": "user", "content": USER_PROMPT}]
    attempts = 0

    try:
        for _ in range(max_turns):
            resp = _create(client, model, effort, messages, tools)
            result.stop_reason = resp.stop_reason
            usage = getattr(resp, "usage", None)
            if usage is not None:
                result.input_tokens += getattr(usage, "input_tokens", 0) or 0
                result.output_tokens += getattr(usage, "output_tokens", 0) or 0

            if resp.stop_reason == "refusal":
                result.error = "Загвар хүсэлтийг татгалзлаа."
                break
            if resp.stop_reason == "max_tokens":
                result.error = "Хариу max_tokens-д хүрч тасарлаа."
                break
            messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason == "pause_turn":
                continue

            uses = [b for b in resp.content if getattr(b, "type", None) == "tool_use"]
            if not uses:
                # Текстээр дуусгасан — хэрэгсэл дуудахыг сануулна
                messages.append({"role": "user", "content":
                                 "Зөвлөмжөө submit_recommendation хэрэгслээр илгээнэ үү."})
                continue

            tool_results = []
            done = False
            for u in uses:
                args = dict(u.input or {})
                if u.name == SUBMIT_TOOL:
                    attempts += 1
                    rec = enforce_confidence(args, tool_confidence)
                    errors = validate_structure(rec)
                    faith = check_recommendation(rec, result.tool_results)
                    result.recommendation, result.structure_errors = rec, errors
                    result.faithfulness = faith
                    if not errors and faith.ok:
                        result.shown = True
                        done = True
                        content, is_error = "Хүлээн авлаа.", False
                    elif attempts >= MAX_SUBMIT_ATTEMPTS:
                        done = True
                        content, is_error = "Хамгаалалтын шалгалт давсангүй.", True
                    else:
                        content = json.dumps({
                            "error": "Зөвлөмж хамгаалалтын шалгалт давсангүй. Засаад дахин илгээ.",
                            "structure_errors": errors,
                            "numbers_not_found_in_tool_results": faith.unmatched,
                        }, ensure_ascii=False)
                        is_error = True
                else:
                    out = call_tool(a, u.name, args)
                    result.tool_calls.append(ToolCall(u.name, args, out))
                    content, is_error = json.dumps(out, ensure_ascii=False), "error" in out
                tool_results.append({"type": "tool_result", "tool_use_id": u.id,
                                     "content": content, "is_error": is_error})
            messages.append({"role": "user", "content": tool_results})
            if done:
                break
        else:
            result.error = f"{max_turns} алхамд зөвлөмж ирсэнгүй."
    except Exception as e:  # noqa: BLE001
        import anthropic

        if isinstance(e, anthropic.AuthenticationError):
            result.error = "API түлхүүр буруу эсвэл тохируулаагүй (ANTHROPIC_API_KEY)."
        elif isinstance(e, anthropic.RateLimitError):
            result.error = "Хүсэлтийн хязгаарт хүрлээ — түр хүлээгээд дахин оролдоно уу."
        elif isinstance(e, (anthropic.APIStatusError, anthropic.APIConnectionError)):
            result.error = f"API алдаа: {e}"
        else:
            raise

    if result.recommendation is not None and not result.shown and not result.error:
        result.error = "Зөвлөмж хамгаалалтын шалгалт давсангүй тул харуулахгүй."
    result.seconds = time.monotonic() - started
    return result
