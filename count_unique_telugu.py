from pathlib import Path
import re
import pandas as pd

root = Path(r'c:/Users/keert/OneDrive - Amrita vishwa vidyapeetham/scriptio continua')
workbook = root / 'Phase 2/Telugu/Complete Dataset/dataset/telugu_sentence_dataset_70_15_15_split.xlsx'

print('Workbook exists:', workbook.exists())
if not workbook.exists():
    raise SystemExit(1)

xl = pd.ExcelFile(workbook)
print('Sheets:', xl.sheet_names)

# Project akshara regex from the Telugu notebooks
AKSHARA_REGEX = re.compile(
    r'[అ-హౘ-ౚ0-9A-Za-z]'
    r'[ఁ-ఃా-ౌౕౖ]*'
    r'(?:్[క-హౘ-ౚ][ఁ-ఃా-ౌౕౖ]*)*'
    r'్?'
    r'[ఁ-ః]?'
    r'|.'
)


def extract_aksharas(text: str):
    words = str(text).split('\u200c')
    out = []
    for word in words:
        if word:
            out.extend(AKSHARA_REGEX.findall(word))
    return out

for sheet in xl.sheet_names:
    df = pd.read_excel(workbook, sheet_name=sheet)
    print('\nSHEET:', sheet)
    print(df.columns.tolist())
    for col in df.columns:
        if 'script' in col.lower() or 'contin' in col.lower() or 'aksh' in col.lower():
            print('Likely script column:', col)
            break

# Find a likely script column in each sheet
for sheet in xl.sheet_names:
    df = pd.read_excel(workbook, sheet_name=sheet)
    script_col = None
    for col in df.columns:
        if 'script' in col.lower() or 'contin' in col.lower() or 'aksh' in col.lower():
            script_col = col
            break
    if script_col is None:
        print(f'No script column in {sheet}')
        continue

    vals = df[script_col].fillna('').astype(str)
    all_tokens = set()
    row_uniques = []
    for val in vals:
        tokens = extract_aksharas(val)
        uniq = set(tokens)
        row_uniques.append(len(uniq))
        all_tokens.update(tokens)
    print(f'\n{sheet}: rows={len(vals)}, unique_tokens_overall={len(all_tokens)}, row_unique_counts_min={min(row_uniques)}, max={max(row_uniques)}, mean={sum(row_uniques)/len(row_uniques):.2f}')
    print('First 10 row unique counts:', row_uniques[:10])

    # first row exact tokens and count
    first_text = str(vals.iloc[0]) if len(vals) else ''
    first_tokens = extract_aksharas(first_text)
    print('First row raw:', first_text[:120])
    print('First row unique count:', len(set(first_tokens)))
    print('First row unique tokens:', sorted(set(first_tokens)))

# compute union across all sheets
union = set()
for sheet in xl.sheet_names:
    df = pd.read_excel(workbook, sheet_name=sheet)
    script_col = None
    for col in df.columns:
        if 'script' in col.lower() or 'contin' in col.lower() or 'aksh' in col.lower():
            script_col = col
            break
    if script_col is None:
        continue
    for val in df[script_col].fillna('').astype(str):
        union.update(extract_aksharas(val))

print('\nUNION ACROSS ALL SHEETS:', len(union))
print(sorted(union)[:200])
