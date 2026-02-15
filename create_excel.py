import pandas as pd
import numpy as np

# Create a sample dataframe
df1 = pd.DataFrame({
    'Category': ['A', 'B', 'C', 'A', 'B'],
    'Values': [10, 20, 30, 15, 25],
    'Date': pd.date_range(start='2023-01-01', periods=5)
})

df2 = pd.DataFrame({
    'Product': ['X', 'Y', 'Z', 'X', 'Y'],
    'Sales': [100, 200, 300, 150, 250],
    'Region': ['North', 'South', 'East', 'West', 'North']
})

# Write to Excel with multiple sheets
with pd.ExcelWriter('test_multisheet.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Summary_Data', index=False)
    df2.to_excel(writer, sheet_name='Detailed_Sales', index=False)

print("Created test_multisheet.xlsx")
