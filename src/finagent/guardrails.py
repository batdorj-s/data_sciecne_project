"""Агентийн гаралтын хамгаалалт (proposal 2.4, DM4).

1. Faithfulness: гаралт дахь мөнгөн дүн, хувь бүр хэрэгслийн үр дүнд байгаа тоотой
   (бөөрөнхийллийн хүлцэлтэйгээр) таарах ёстой. Таарахгүй тоо байвал гаралтыг харуулахгүй.
2. Бүтэц: зөвлөмж нь `RECOMMENDATION_SCHEMA`-ийн дагуу байна (тактик бүр KPI, ₮ нөлөөтэй).
3. Итгэлийн түвшин: хэрэгслүүд "low" гэсэн бол агент "low"-оос дээш бичиж болохгүй.

LLM-ийн дуудлага өөрөө (Sprint 4) энэ модульд хамааралгүй — энд зөвхөн шалгалт байна.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field

# "61.7 сая ₮", "1,763,500 ₮", "20.3%", "−14.9 сая", "(14.9)"
_NUM_RE = re.compile(
    r"(?P<sign>[-−])?\(?(?P<num>\d{1,3}(?:[ ,]\d{3})+|\d+(?:[.,]\d+)?)\)?\s*"
    r"(?P<unit>сая|тэрбум|мянга|%|₮)?",
)
_SCALE = {"сая": 1e6, "тэрбум": 1e9, "мянга": 1e3}

RECOMMENDATION_SCHEMA = {
    "type": "object",
    "required": ["trend", "confidence", "drivers", "risks", "tactics", "data_warnings"],
    "properties": {
        "trend": {"type": "string", "enum": ["growth", "stable", "decline", "stable_growth",
                                             "stable_decline"]},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "summary": {"type": "string"},
        "drivers": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}},
        "tactics": {
            "type": "array", "minItems": 3, "maxItems": 6,
            "items": {
                "type": "object",
                "required": ["action", "kpi", "impact_mnt", "priority"],
                "properties": {
                    "action": {"type": "string"},
                    "kpi": {"type": "string"},
                    "impact_mnt": {"type": "number"},
                    "priority": {"type": "string", "enum": ["high", "medium", "low"]},
                },
            },
        },
        "data_warnings": {"type": "array", "items": {"type": "string"}},
    },
}


@dataclass
class FaithfulnessReport:
    checked: int
    matched: int
    unmatched: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return 1.0 if self.checked == 0 else self.matched / self.checked

    @property
    def ok(self) -> bool:
        return not self.unmatched


def _walk_numbers(obj) -> Iterable[float]:
    if isinstance(obj, bool):
        return
    if isinstance(obj, (int, float)):
        yield float(obj)
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _walk_numbers(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk_numbers(v)


def _parse_num(raw: str) -> float:
    if re.fullmatch(r"\d{1,3}(?:[ ,]\d{3})+", raw):
        return float(raw.replace(",", "").replace(" ", ""))
    return float(raw.replace(",", "."))


def extract_numbers(text: str) -> list[tuple[str, float, str | None]]:
    """(эх бичвэр, утга, нэгж) жагсаалт.

    Нэгжгүй тоо нь дараагийн тооноос "–" эсвэл "→"-ээр тусгаарлагдсан бол нэгжийг өвлөнө
    ("61.7–86.8 сая", "10.8 → 12.5 сая"). Бусад нэгжгүй тоог (огноо, дугаар) алгасна.
    """
    found = list(_NUM_RE.finditer(text))
    units: list[str | None] = [m.group("unit") for m in found]
    for i in range(len(found) - 2, -1, -1):
        if units[i] is None and units[i + 1] is not None:
            gap = text[found[i].end():found[i + 1].start()]
            if re.fullmatch(r"\s*[–→-]\s*", gap) or gap == "":
                units[i] = units[i + 1]
    out = []
    for m, unit in zip(found, units, strict=True):
        if unit is None:
            continue
        val = _parse_num(m.group("num")) * _SCALE.get(unit, 1.0)
        neg = bool(m.group("sign")) or m.group(0).startswith("(")
        out.append((m.group(0).strip(), -val if neg else val, unit))
    return out


def _tolerance(raw: str | None, unit: str | None) -> float:
    """Бичсэн нарийвчлалаас хамаарсан хүлцэл: "28%" → ±0.5 нэгж, "61.7 сая" → ±0.05 сая."""
    m = re.search(r"\d+[.,](\d+)", raw or "")
    dec = len(m.group(1)) if m and not re.search(r"\d{1,3}(?:[ ,]\d{3})+", raw or "") else 0
    half = 0.5 * 10 ** (-dec) + 1e-9
    if unit == "%":
        return half
    if unit in _SCALE:
        return half * _SCALE[unit]
    return 1.0


def _matches(value: float, unit: str | None, ref: list[float], raw: str | None = None) -> bool:
    tol = _tolerance(raw, unit)
    if unit == "%":
        return any(abs(abs(value) - abs(r) * 100) <= tol or abs(abs(value) - abs(r)) <= tol
                   for r in ref)
    return any(abs(abs(value) - abs(r)) <= tol for r in ref)


def check_faithfulness(text: str, tool_results: Iterable[dict]) -> FaithfulnessReport:
    ref = list({n for r in tool_results for n in _walk_numbers(r)})
    # Хэрэгслийн тооны зөрүү/нийлбэрийг агент ихэвчлэн бичдэг тул хос зөрүүг нэмнэ
    big = [r for r in ref if abs(r) >= 1000]
    ref += [a - b for a in big for b in big if a != b]
    nums = extract_numbers(text)
    unmatched = [raw for raw, val, unit in nums if not _matches(val, unit, ref, raw)]
    return FaithfulnessReport(checked=len(nums), matched=len(nums) - len(unmatched),
                              unmatched=unmatched)


def check_recommendation(rec: dict, tool_results: list[dict]) -> FaithfulnessReport:
    """Зөвлөмжийн текст дэх тоо + тактик бүрийн impact_mnt-ийг хэрэгслийн үр дүнтэй тулгана."""
    texts = []
    for key in ("summary", "drivers", "risks", "data_warnings"):
        v = rec.get(key)
        texts += v if isinstance(v, list) else ([v] if v else [])
    for t in rec.get("tactics", []):
        texts += [str(t.get("action", "")), str(t.get("kpi", ""))]
    rep = check_faithfulness("\n".join(texts), tool_results)
    ref = list({n for r in tool_results for n in _walk_numbers(r)})
    for i, t in enumerate(rec.get("tactics", []), 1):
        impact = t.get("impact_mnt")
        if isinstance(impact, (int, float)):
            rep.checked += 1
            if _matches(float(impact), None, ref) or _matches(float(impact), "сая", ref, "0.0"):
                rep.matched += 1
            else:
                rep.unmatched.append(f"тактик {i} impact_mnt={impact}")
    return rep


def validate_structure(rec: dict) -> list[str]:
    """JSON schema-ийн гол шаардлагыг (хамааралгүйгээр) шалгана; алдааны жагсаалт буцаана."""
    errors = []
    for key in RECOMMENDATION_SCHEMA["required"]:
        if key not in rec:
            errors.append(f"'{key}' талбар алга")
    tactics = rec.get("tactics", [])
    if not 3 <= len(tactics) <= 6:
        errors.append(f"тактик 3–6 байх ёстой (одоо {len(tactics)})")
    for i, t in enumerate(tactics, 1):
        for key in ("action", "kpi", "impact_mnt", "priority"):
            if key not in t:
                errors.append(f"тактик {i}: '{key}' алга")
    if rec.get("confidence") not in (None, "low", "medium", "high"):
        errors.append("confidence буруу")
    return errors


def enforce_confidence(rec: dict, tool_confidence: str) -> dict:
    """Агент хэрэгслийн итгэлийн түвшнээс дээш бичсэн бол бууруулна."""
    order = {"low": 0, "medium": 1, "high": 2}
    if order.get(rec.get("confidence", "low"), 0) > order[tool_confidence]:
        rec = {**rec, "confidence": tool_confidence}
    return rec


