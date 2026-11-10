"""Нэгдсэн гүйлгээний бүтэц ба ангиллын схем.

Бүх модуль энд тодорхойлсон баганын нэр, мөрийн түлхүүрийг (line key) ашиглана.
Эх файлын ангилал компани бүрт өөр байж болох тул `classify` модуль тэдгээрийг
энэ схемд буулгадаг.
"""

from __future__ import annotations

from enum import StrEnum

# --- Эх файлын баганууд (Data Dictionary) -----------------------------------------
RAW_COLUMNS = {
    "Өдөр": "date_raw",
    "Харилцагч": "counterparty",
    "Дүн": "amount_raw",
    "Ангилал": "src_category",
    "Дэд ангилал": "src_subcategory",
    "Тайлбар": "description",
}
REQUIRED_RAW = ["Өдөр", "Харилцагч", "Дүн", "Ангилал"]


# --- Нэгдсэн бүтцийн баганууд ------------------------------------------------------
class Col(StrEnum):
    TXN_ID = "txn_id"
    DATE = "date"
    MONTH = "month"  # "YYYY-MM"
    COUNTERPARTY = "counterparty"
    AMOUNT = "amount"  # үргэлж эерэг
    DIRECTION = "direction"  # "in" | "out"
    SIGNED = "signed_amount"  # орлого +, зарлага −
    SRC_CATEGORY = "src_category"
    SRC_SUBCATEGORY = "src_subcategory"
    DESCRIPTION = "description"
    ACTIVITY = "activity"
    LINE = "line"
    CLASS_SOURCE = "class_source"  # "rule" | "manual" | "model"
    SOURCE_FILE = "source_file"
    SOURCE_ROW = "source_row"


TXN_COLUMNS = [c.value for c in Col]


class Activity(StrEnum):
    OPERATING = "operating"  # үндсэн үйл ажиллагаа (ашигт тооцогдоно)
    FINANCING = "financing"  # зээл, эздийн зээл (ашигт тооцогдохгүй)
    INTERNAL = "internal"  # өөрийн данс хооронд (банк ↔ касс)
    UNCLASSIFIED = "unclassified"


class Line(StrEnum):
    # Үйл ажиллагааны орлого
    SALES = "sales"
    OTHER_INCOME = "other_income"
    # Үйл ажиллагааны зардал
    PAYROLL = "payroll"
    PROJECT = "project"
    OFFICE = "office"
    RENT = "rent"
    OPEX = "opex"
    # Санхүүгийн үйл ажиллагаа
    LOAN_REPAYMENT = "loan_repayment"
    OWNER_LOAN_IN = "owner_loan_in"
    LOAN_RETURN_IN = "loan_return_in"
    PASSTHROUGH_LOAN_IN = "passthrough_loan_in"
    PASSTHROUGH_LOAN_OUT = "passthrough_loan_out"
    # Дотоод шилжүүлэг
    TRANSFER_TO_CASH = "transfer_to_cash"
    TRANSFER_FROM_CASH = "transfer_from_cash"
    # Ангилагдаагүй
    UNCLASSIFIED_IN = "unclassified_in"
    UNCLASSIFIED_OUT = "unclassified_out"


LINE_LABELS: dict[str, str] = {
    Line.SALES: "Борлуулалтын орлого",
    Line.OTHER_INCOME: "Бусад орлого (дэд түрээс г.м.)",
    Line.PAYROLL: "Цалингийн зардал",
    Line.PROJECT: "Төслийн зардал",
    Line.OFFICE: "Оффисын зардал",
    Line.RENT: "Түрээсийн зардал",
    Line.OPEX: "Үйл ажиллагааны бусад зардал (ҮАЗ)",
    Line.LOAN_REPAYMENT: "Зээлийн төлбөр (гарсан)",
    Line.OWNER_LOAN_IN: "Эздийн зээл (орсон)",
    Line.LOAN_RETURN_IN: "Зээлийн буцаалт (орсон)",
    Line.PASSTHROUGH_LOAN_IN: "Дамжин өнгөрсөн зээл (орсон)",
    Line.PASSTHROUGH_LOAN_OUT: "Дамжин өнгөрсөн зээл (гарсан)",
    Line.TRANSFER_TO_CASH: "Дотоод шилжүүлэг (банк → касс)",
    Line.TRANSFER_FROM_CASH: "Дотоод шилжүүлэг (касс → банк)",
    Line.UNCLASSIFIED_IN: "Ангилагдаагүй орлого",
    Line.UNCLASSIFIED_OUT: "Ангилагдаагүй зарлага",
}

LINE_ACTIVITY: dict[str, Activity] = {
    Line.SALES: Activity.OPERATING,
    Line.OTHER_INCOME: Activity.OPERATING,
    Line.PAYROLL: Activity.OPERATING,
    Line.PROJECT: Activity.OPERATING,
    Line.OFFICE: Activity.OPERATING,
    Line.RENT: Activity.OPERATING,
    Line.OPEX: Activity.OPERATING,
    Line.LOAN_REPAYMENT: Activity.FINANCING,
    Line.OWNER_LOAN_IN: Activity.FINANCING,
    Line.LOAN_RETURN_IN: Activity.FINANCING,
    Line.PASSTHROUGH_LOAN_IN: Activity.FINANCING,
    Line.PASSTHROUGH_LOAN_OUT: Activity.FINANCING,
    Line.TRANSFER_TO_CASH: Activity.INTERNAL,
    Line.TRANSFER_FROM_CASH: Activity.INTERNAL,
    Line.UNCLASSIFIED_IN: Activity.UNCLASSIFIED,
    Line.UNCLASSIFIED_OUT: Activity.UNCLASSIFIED,
}

INCOME_LINES = [Line.SALES, Line.OTHER_INCOME]
EXPENSE_LINES = [Line.PAYROLL, Line.PROJECT, Line.OFFICE, Line.RENT, Line.OPEX]
FINANCING_LINES = [
    Line.LOAN_REPAYMENT,
    Line.OWNER_LOAN_IN,
    Line.LOAN_RETURN_IN,
    Line.PASSTHROUGH_LOAN_IN,
    Line.PASSTHROUGH_LOAN_OUT,
]


def line_label(line: str) -> str:
    return LINE_LABELS.get(line, line)
