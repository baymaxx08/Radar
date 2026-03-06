import csv

filepath = r"c:\Users\Siddhartha Reddy\Desktop\radar\20250319\202503019_GH_exposed_corn_ear_cls001.csv"

with open(filepath, 'r') as f:
    reader = csv.reader(f)
    rows = list(reader)

print(f"Total rows: {len(rows)}")
print(f"\nRow 2 length: {len(rows[2])}")
print(f"Row 2 first 5:")
for i in range(5):
    print(f"  [{i}] = '{rows[2][i]}'")

print(f"\n'MrmFullScanInfo' in row[2]: {'MrmFullScanInfo' in rows[2]}")
print(f"' MrmFullScanInfo' in row[2]: {' MrmFullScanInfo' in rows[2]}")

# Try with stripped values
print("\nSearching with stripped values...")
for i in range(3):
    row = rows[i]
    if any('MrmFullScanInfo' in col.strip() for col in row):
        print(f"Row {i} has MrmFullScanInfo (after strip)")





