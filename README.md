# Санхүүгийн агент AI

ЖДБ сар бүрийн банкны гүйлгээгээ оруулахад систем **тайлан**, **дашбоард**-ыг автоматаар гаргаж,
сүүлийн 3 сар (улирал) дээр тулгуурлан **дараа сарын таамгийн хүрээ** болон **тактик**-ийг санал болгоно.

> Агент тоо зохиохгүй: бүх тооцоог детерминист хэрэгслүүд (`finagent.tools`) хийж,
> агент нь зөвхөн тэдгээрийн JSON үр дүнг ашиглан дүгнэлт бичнэ.

Төслийн төрөл: Data Science Project Proposal — CRISP-DM. Пилот: Компани А, 2026 оны I улирал, 185 гүйлгээ.

## Архитектур

```
 ┌──────────── Детерминист давхарга (src/finagent) ────────────┐   ┌──── Агентийн давхарга ────┐
 │ ingest → classify → checks (R1–R9) → reports → forecast     │ → │ tools (6) → AI агент (S4)  │
 └──────────────────────────────────────────────────────────────┘   └────────────────────────────┘
                         ▲                                                    │
                    app/ (Streamlit вэб: файл оруулах, тайлан, дашбоард, таамаг, сценари)
```

| Модуль | Үүрэг |
|---|---|
| `finagent.schema` | Нэгдсэн гүйлгээний бүтэц, ангиллын схем |
| `finagent.ingest` | Сарын хуулга (`.xlsx`) унших, баганыг шалгах |
| `finagent.classify` | Эх ангиллыг нэгдсэн схемд буулгах (ҮА / санхүүгийн / дотоод) |
| `finagent.checks` | R1–R9 шалгалтын дүрэм, үлдэгдлийн тулгалт |
| `finagent.reports` | Орлого-зарлага, мөнгөн урсгал, зардлын задаргаа |
| `finagent.forecast` | Жинлэсэн дундаж + чиг хандлагын хүрээ, сценари |
| `finagent.tools` | Агентийн 6 хэрэгсэл (`get_quarter_overview` …) |
| `finagent.anonymize` | Харилцагчийн нэрийг кодоор солих |
| `finagent.pipeline` | `Analysis` — апп ба агент хуваалцах нэг тооцоо |
| `finagent.guardrails` | Faithfulness, зөвлөмжийн бүтэц, итгэлийн түвшин |
| `finagent.export` | Excel тайлан |
| `finagent.agent` | DM4: Claude tool use агент (`claude-opus-5-5`) |
| `finagent.store` | Хадгалалт (SQLite / PostgreSQL), таамаг ↔ бодит дүн, үнэлгээ |
| `finagent.ml` | DM1 ангилагч, DM2 Isolation Forest + алдаа суулгах |

Вэб апп (`app/`): **Өгөгдөл → Дашбоард → Тайлан → Шалгалт ба ангилал → Таамаг ба сценари → AI агент**.

Дэлгэрэнгүй: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — шийдвэрүүд, R1–R9, пилотын тоонуудын тулгалт.

## Эхлүүлэх

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[web,agent,dev]"
export ANTHROPIC_API_KEY=...        # AI агентэд (заавал биш)
export DATABASE_URL=postgresql+psycopg://...   # заавал биш; анхдагч: data/finagent.sqlite
streamlit run app/Home.py
```

ML загваруудын үнэлгээ:

```bash
python -m finagent.ml.classifier
```

Тест:

```bash
pytest && ruff check .
```

## Өгөгдөл ба нууцлал

- `data/raw/` — бодит хуулга (хувь хүний нэртэй). **Git-д ордоггүй.**
- `data/sample/` — `scripts/anonymize_raw.py`-аар харилцагчийн нэрийг `CP-001` хэлбэрийн кодоор
  сольсон пилот өгөгдөл. Апп анхдагчаар үүнийг ачаална.

```bash
python scripts/anonymize_raw.py path/to/sar_1.xlsx path/to/sar_2.xlsx path/to/sar_3.xlsx
```

Тайлан нь **мөнгөн суурьтай удирдлагын тайлан** бөгөөд НББОУС-ийн санхүүгийн тайланг орлохгүй.

## Sprint төлөвлөгөө

| Sprint | Агуулга | Төлөв |
|---|---|---|
| 1 | Бизнесийн ойлголт, ангиллын схем | ✅ суурь схем |
| 2 | Өгөгдөл, EDA | ✅ банкны экспорт, хадгалалт · ⏳ 1,000+ шошго |
| 3 | DM1 ангилал, DM2 шалгалт | ✅ R1–R9, TF-IDF+LogReg, Isolation Forest |
| 4 | DM3 таамаг, DM4 агент | ✅ суурь таамаг, Claude агент, guardrails · ⏳ ETS (6+ сар) |
| 5 | Үнэлгээ | ✅ MAPE/coverage, үнэлгээний UI, алдаа суулгах · ⏳ бодит хэрэглэгчийн үнэлгээ |
| 6 | Вэб MVP | ✅ эхний хувилбар |
