"""Нууцлал: харилцагчийн нэрийг кодоор солих (proposal 2.4, 3.2).

- Харилцагч бүрт тогтмол код (`CP-001`) өгнө — эхэлж гарсан дарааллаар.
- Тайлбар дотор орсон харилцагчийн нэрийн үгсийг `***` болгоно
  (жишээ: "ZURAG AWALT BENZIN,АДИЛБИШ ЖАМЪЯН" → "ZURAG AWALT BENZIN,*** ***").
- Нэмэлт үгсийг (компанийн өөрийн нэр г.м.) `extra_terms`-ээр солино.

Агент руу илгээх өгөгдөл ба репо дахь жишээ өгөгдөл заавал энэ шатыг дамжина.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

import pandas as pd

# Байгууллагын хэлбэрийг заадаг, нэр биш үгс — тайлбараас устгахгүй
_ORG_WORDS = {"ХХК", "ХК", "ТББ", "ББСБ", "БАНК", "САЛБАР", "ИХ", "СУРГУУЛЬ", "КОО", "НҮ"}
_MIN_TOKEN = 3


def _name_tokens(name: str) -> set[str]:
    words = re.findall(r"[А-ЯЁӨҮа-яёөүA-Za-z]+", name or "")
    return {w.upper() for w in words if len(w) >= _MIN_TOKEN and w.upper() not in _ORG_WORDS}


_TRANSLIT = {
    "А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E", "Ё": "YO", "Ж": "J", "З": "Z",
    "И": "I", "Й": "I", "К": "K", "Л": "L", "М": "M", "Н": "N", "О": "O", "Ө": "U", "П": "P",
    "Р": "R", "С": "S", "Т": "T", "У": "U", "Ү": "U", "Ф": "F", "Х": "H", "Ц": "TS", "Ч": "CH",
    "Ш": "SH", "Щ": "SH", "Ъ": "", "Ы": "II", "Ь": "I", "Э": "E", "Ю": "YU", "Я": "YA",
}


def transliterate(word: str) -> set[str]:
    """Монгол кирилл нэрийн түгээмэл латин хувилбарууд (НАРАНЗОН → NARANZON)."""
    base = "".join(_TRANSLIT.get(ch, ch) for ch in word.upper())
    variants = {base}
    # Ү/Ө-г хуулгад V/O гэж бичих нь элбэг
    alt = "".join({"Ү": "V", "Ө": "O"}.get(ch, _TRANSLIT.get(ch, ch)) for ch in word.upper())
    variants.add(alt)
    return {v for v in variants if len(v) >= _MIN_TOKEN + 1}


def is_organization(name: str | None) -> bool:
    """Байгууллагын нэр (ХХК, БАНК г.м.) хувь хүний мэдээлэлд тооцогдохгүй."""
    words = {w.upper() for w in re.findall(r"[А-ЯЁӨҮа-яёөүA-Za-z]+", name or "")}
    return bool(words & _ORG_WORDS)


def build_codebook(counterparties: Iterable[str], prefix: str = "CP") -> dict[str, str]:
    book: dict[str, str] = {}
    for cp in counterparties:
        if cp and cp not in book:
            book[cp] = f"{prefix}-{len(book) + 1:03d}"
    return book


def scrub_text(text: str | None, tokens: set[str], replacement: str = "***") -> str | None:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return text
    if not tokens:
        return text

    def _repl(m: re.Match) -> str:
        word = m.group(0).upper()
        # Банкны хуулга тайлбарыг тасалдаг ("БИЛГҮҮН" → "БИЛГҮҮ") тул угтвараар ч таарна
        if word in tokens or (
            len(word) >= _MIN_TOKEN and any(t.startswith(word) for t in tokens)
        ):
            return replacement
        return m.group(0)

    latin = {v for t in tokens for v in transliterate(t)}

    def _repl_latin(m: re.Match) -> str:
        return replacement if m.group(0).upper() in latin else m.group(0)

    text = re.sub(r"[А-ЯЁӨҮа-яёөү]+", _repl, text)
    return re.sub(r"[A-Za-z]+", _repl_latin, text)


def anonymize_frame(
    df: pd.DataFrame,
    codebook: dict[str, str] | None = None,
    extra_terms: dict[str, str] | None = None,
    counterparty_col: str = "counterparty",
    description_col: str = "description",
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Хуулбар DataFrame болон ашигласан кодын дэвтрийг буцаана."""
    out = df.copy()
    codebook = dict(codebook or {})
    for cp in out[counterparty_col].dropna():
        if cp not in codebook:
            codebook[cp] = f"CP-{len(codebook) + 1:03d}"

    # Тайлбараас зөвхөн хувь хүний нэрийг нууна; байгууллагын нэр (Chimege г.м.) үлдэнэ
    tokens: set[str] = set()
    for cp in codebook:
        if not is_organization(cp):
            tokens |= _name_tokens(cp)

    desc = out[description_col]
    for term, repl in (extra_terms or {}).items():
        desc = desc.map(
            lambda s, t=term, r=repl: re.sub(re.escape(t), r, s, flags=re.IGNORECASE)
            if isinstance(s, str) else s
        )
    out[description_col] = desc.map(lambda s: scrub_text(s, tokens))
    out[counterparty_col] = out[counterparty_col].map(lambda v: codebook.get(v, v))
    return out, codebook
