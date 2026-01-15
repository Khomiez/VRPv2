# -*- coding: utf-8 -*-
import pandas as pd
import openpyxl
import json

excel_file = 'd:/projects/python/VRPv2/distance_matrix_full_138_zones-edit4.xlsx'

wb = openpyxl.load_workbook(excel_file)
sheet_names = wb.sheetnames

analysis = {}

for name in sheet_names:
    print(f'\nAnalyzing sheet: {repr(name)}')
    df = pd.read_excel(excel_file, sheet_name=name)

    sheet_info = {
        'name': name,
        'shape': df.shape,
        'columns': list(df.columns),
        'first_rows': df.head(3).to_dict('records'),
        'data_types': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'missing_values': df.isnull().sum().to_dict(),
        'numeric_summary': {}
    }

    # Get numeric column statistics
    numeric_cols = df.select_dtypes(include=['number']).columns
    for col in numeric_cols[:10]:  # First 10 numeric columns
        sheet_info['numeric_summary'][col] = {
            'min': float(df[col].min()) if pd.notna(df[col].min()) else None,
            'max': float(df[col].max()) if pd.notna(df[col].max()) else None,
            'mean': float(df[col].mean()) if pd.notna(df[col].mean()) else None
        }

    analysis[name] = sheet_info

    print(f'  Shape: {df.shape}')
    print(f'  Columns: {len(df.columns)}')

# Save analysis to JSON
with open('d:/projects/python/VRPv2/excel_analysis.json', 'w', encoding='utf-8') as f:
    json.dump(analysis, f, indent=2, ensure_ascii=False)

print('\nAnalysis saved to excel_analysis.json')

# Also print detailed info for the largest sheet
print('\n' + '='*80)
print('DETAILED ANALYSIS OF LARGEST SHEET')
print('='*80)

largest_sheet = max(analysis.keys(), key=lambda x: analysis[x]['shape'][1])
print(f'\nLargest sheet: {repr(largest_sheet)}')
print(f'Shape: {analysis[largest_sheet]["shape"]}')

df_large = pd.read_excel(excel_file, sheet_name=largest_sheet)

print(f'\nFirst 10 column names:')
for i, col in enumerate(df_large.columns[:10]):
    print(f'  {i}: {col}')

print(f'\nLast 10 column names:')
for i, col in enumerate(df_large.columns[-10:]):
    print(f'  {len(df_large.columns)-10+i}: {col}')

print(f'\nFirst 5 rows:')
print(df_large.head())

# Check for depot
print('\n' + '='*80)
print('CHECKING FOR DEPOT')
print('='*80)

# Look for variations of "จุดทิ้ง" in the data
for name in sheet_names:
    df = pd.read_excel(excel_file, sheet_name=name)
    print(f'\nSheet {repr(name)}:')

    # Check Destination column
    if 'Destination' in df.columns:
        dest_values = df['Destination'].dropna().unique()
        print(f'  Unique Destination values ({len(dest_values)}):')
        for val in dest_values[:min(20, len(dest_values))]:
            print(f'    - {val}')

    # Check for any column containing depot-like values
    for col in df.columns:
        if df[col].dtype == 'object':
            unique_vals = df[col].dropna().unique()
            for val in unique_vals:
                val_str = str(val)
                if 'จุด' in val_str or 'ทิ้ง' in val_str or 'depot' in val_str.lower():
                    print(f'  Found depot-like value in column "{col}": {val}')
