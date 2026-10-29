from collections import Counter
from pathlib import Path
import re

import pandas as pd
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.worksheet.table import Table, TableStyleInfo


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "Phase 2" / "Telugu" / "Complete Dataset" / "dataset" / "telugu_sentence_dataset_70_15_15_split.xlsx"
OUTPUT = ROOT / "Phase 2" / "Telugu" / "Complete Dataset" / "telugu_akshara_frequencies.xlsx"

# The tokenizer used in the Telugu training notebooks.
AKSHARA_REGEX = re.compile(
    r"[అ-హౘ-ౚ0-9A-Za-z]"
    r"[ఁ-ఃా-ౌౕౖ]*"
    r"(?:్[క-హౘ-ౚ][ఁ-ఃా-ౌౕౖ]*)*"
    r"్?"
    r"[ఁ-ః]?"
    r"|."
)


def tokens(text: object) -> list[str]:
    return [
        token
        for part in str(text).split("\u200c")
        if part
        for token in AKSHARA_REGEX.findall(part)
    ]


def frequency_table(values: pd.Series) -> Counter:
    counts: Counter = Counter()
    for value in values.dropna():
        counts.update(tokens(value))
    return counts


def write_sheet(workbook: Workbook, name: str, counts: Counter) -> None:
    sheet = workbook.create_sheet(name)
    sheet.append(["Character", "Frequency"])
    for character, frequency in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        sheet.append([character, frequency])

    header = sheet["A1:B1"]
    for row in header:
        for cell in row:
            cell.fill = PatternFill("solid", fgColor="1F4E78")
            cell.font = Font(bold=True, color="FFFFFF", name="Nirmala UI")
            cell.alignment = Alignment(horizontal="center")

    for row in sheet.iter_rows(min_row=2, max_col=2):
        row[0].font = Font(name="Nirmala UI")
        row[0].alignment = Alignment(horizontal="left")
        row[1].alignment = Alignment(horizontal="right")

    last_row = sheet.max_row
    table = Table(displayName=f"{name}Frequency", ref=f"A1:B{last_row}")
    table.tableStyleInfo = TableStyleInfo(
        name="TableStyleMedium2", showFirstColumn=False, showLastColumn=False,
        showRowStripes=True, showColumnStripes=False,
    )
    sheet.add_table(table)
    sheet.freeze_panes = "A2"
    sheet.column_dimensions["A"].width = 20
    sheet.column_dimensions["B"].width = 14


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source workbook not found: {SOURCE}")

    workbook = Workbook()
    workbook.remove(workbook.active)
    all_counts: Counter = Counter()

    for source_sheet, output_sheet in (
        ("Training", "Training"),
        ("Testing", "Testing"),
        ("Validation", "Validation"),
    ):
        frame = pd.read_excel(SOURCE, sheet_name=source_sheet)
        counts = frequency_table(frame["scriptio_continua"])
        all_counts.update(counts)
        write_sheet(workbook, output_sheet, counts)

    write_sheet(workbook, "Union", all_counts)
    workbook.save(OUTPUT)

    # Re-open the artifact to confirm all requested sheets and row counts exist.
    verified = load_workbook(OUTPUT, read_only=True, data_only=True)
    expected = {"Training", "Testing", "Validation", "Union"}
    if set(verified.sheetnames) != expected:
        raise RuntimeError(f"Unexpected output sheets: {verified.sheetnames}")
    print(f"Created: {OUTPUT}")
    for sheet_name in verified.sheetnames:
        print(f"{sheet_name}: {verified[sheet_name].max_row - 1} characters")


if __name__ == "__main__":
    main()
