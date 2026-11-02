# Phase 1 Automation and Helper Scripts

This directory contains automation utilities and data transformation scripts supporting Phase 1 and cross-phase evaluation.

---

## Scripts

### 1. `add_avg_score_sheet.ps1`
- **Language**: PowerShell
- **Purpose**: Automates adding an aggregate summary worksheet containing mean performance scores across model runs directly into existing Excel results workbooks.

### 2. `create_telugu_akshara_frequency_workbook.py`
- **Language**: Python
- **Dependencies**: `pandas`, `openpyxl`
- **Purpose**: Extracts aksharas from Telugu dataset files using phonetic syllabic regex rules and compiles corpus-wide frequency statistics into a standardized Excel workbook for linguistic complexity analysis.
