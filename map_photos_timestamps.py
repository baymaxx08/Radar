import os
import json
from datetime import datetime
from pathlib import Path
import csv
import struct

# pip install Pillow if not already installed
try:
    from PIL import Image as PILImage
    from PIL.ExifTags import TAGS
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    print("Note: Pillow not installed. Using basic metadata only.")

data_folder = r"c:\Users\Siddhartha Reddy\Desktop\radar\20250319"

def get_exif_datetime(image_path):
    """Extract datetime from EXIF metadata if available"""
    if not HAS_PIL:
        return None
    
    try:
        image = PILImage.open(image_path)
        exif_data = image._getexif()
        
        if exif_data is None:
            return None
        
        for tag_id, value in exif_data.items():
            tag_name = TAGS.get(tag_id, tag_id)
            
            # Look for DateTimeOriginal (36867) or DateTime (306)
            if tag_name in ['DateTimeOriginal', 'DateTime']:
                try:
                    return datetime.strptime(value, '%Y:%m:%d %H:%M:%S')
                except:
                    pass
    except Exception as e:
        pass
    
    return None

# Get all image files
image_files = []
for img_file in os.listdir(data_folder):
    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
        image_files.append(img_file)

# Get image file timestamps with EXIF data
image_timestamps = {}
for img_file in sorted(image_files):
    img_path = os.path.join(data_folder, img_file)
    
    # Try EXIF first
    exif_time = get_exif_datetime(img_path)
    
    # Fall back to file modification time
    mod_time = os.path.getmtime(img_path)
    mod_datetime = datetime.fromtimestamp(mod_time)
    
    # Use EXIF if available, otherwise use file modification time
    if exif_time:
        primary_time = exif_time
        time_source = "EXIF"
    else:
        primary_time = mod_datetime
        time_source = "File Modified"
    
    image_timestamps[img_file] = {
        'file_path': img_path,
        'timestamp_readable': primary_time.isoformat(),
        'timestamp_display': primary_time.strftime('%Y-%m-%d %H:%M:%S'),
        'time_source': time_source,
        'file_modified': mod_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        'exif_datetime': exif_time.strftime('%Y-%m-%d %H:%M:%S') if exif_time else 'No EXIF data'
    }

# Get CSV timestamps
csv_data = {}
csv_files = sorted([f for f in os.listdir(data_folder) if f.endswith('.csv')])

for csv_file in csv_files:
    csv_path = os.path.join(data_folder, csv_file)
    timestamps = []
    
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            try:
                if row and row[0].strip():
                    val = row[0].strip()
                    if val.replace('.', '', 1).lstrip('-').isdigit():
                        ts = float(val)
                        ts_datetime = datetime.fromtimestamp(ts)
                        timestamps.append({
                            'unix': ts,
                            'readable': ts_datetime.strftime('%Y-%m-%d %H:%M:%S.%f')[:-3],
                            'display': ts_datetime.strftime('%Y-%m-%d %H:%M:%S')
                        })
            except (ValueError, IndexError):
                pass
    
    if timestamps:
        csv_data[csv_file] = {
            'min_timestamp': min(timestamps, key=lambda x: x['unix']),
            'max_timestamp': max(timestamps, key=lambda x: x['unix']),
            'total_records': len(timestamps),
            'sample_timestamps': timestamps[:5]
        }

# Create mapping report
mapping_report = {
    'metadata': {
        'report_generated': datetime.now().isoformat(),
        'data_folder': data_folder,
        'images_found': len(image_files),
        'csv_files_found': len(csv_files)
    },
    'images': image_timestamps,
    'csv_data': csv_data,
    'image_to_csv_closest': []
}

# Find closest CSV timestamps for each image
all_csv_times = []
for csv_file, csv_info in csv_data.items():
    if 'sample_timestamps' in csv_info:
        for ts in csv_info['sample_timestamps']:
            all_csv_times.append({
                'source_file': csv_file,
                'unix': ts['unix'],
                'readable': ts['readable']
            })

for img, img_data in sorted(image_timestamps.items()):
    try:
        img_readable = img_data['timestamp_readable']
        img_dt = datetime.fromisoformat(img_readable)
        
        # Find closest match in CSV
        if all_csv_times:
            closest = min(all_csv_times, 
                         key=lambda x: abs(datetime.fromisoformat(x['readable']) - img_dt).total_seconds())
            time_diff = abs(datetime.fromisoformat(closest['readable']) - img_dt).total_seconds()
            
            mapping_report['image_to_csv_closest'].append({
                'image': img,
                'image_timestamp': img_data['timestamp_display'],
                'time_source': img_data['time_source'],
                'closest_csv_file': closest['source_file'],
                'closest_csv_timestamp': closest['readable'],
                'time_difference_seconds': time_diff
            })
    except:
        pass

# Sort by time difference
mapping_report['image_to_csv_closest'].sort(key=lambda x: x['time_difference_seconds'])

# Print results
print("=" * 90)
print("PHOTO TO TIMESTAMP MAPPING REPORT")
print("=" * 90)
print()

print(f"Images Found: {len(image_files)}")
print(f"CSV Files Found: {len(csv_files)}")
print()

print("-" * 90)
print("IMAGE TIMESTAMPS:")
print("-" * 90)
for img, data in sorted(image_timestamps.items()):
    print(f"  📷 {img}")
    print(f"     Primary Timestamp:  {data['timestamp_display']} (Source: {data['time_source']})")
    print(f"     File Modified:      {data['file_modified']}")
    print(f"     EXIF DateTime:      {data['exif_datetime']}")
    print()

print("-" * 90)
print("CSV DATA SUMMARY:")
print("-" * 90)
for csv_file, info in sorted(csv_data.items()):
    print(f"  📊 {csv_file}")
    print(f"     Records Found: {info['total_records']}")
    print(f"     Time Range: {info['min_timestamp']['readable']} to {info['max_timestamp']['readable']}")
    print()

print("-" * 90)
print("IMAGE TO CLOSEST CSV TIMESTAMP MAPPING:")
print("-" * 90)
for mapping in mapping_report['image_to_csv_closest']:
    print(f"  📷 {mapping['image']}")
    print(f"     Image Time:        {mapping['image_timestamp']}")
    print(f"     Closest CSV Entry: {mapping['closest_csv_timestamp']}")
    print(f"     Source CSV File:   {mapping['closest_csv_file']}")
    print(f"     Time Difference:   {mapping['time_difference_seconds']:.1f} seconds")
    print()

# Save report to JSON file
report_path = os.path.join(data_folder, "photo_timestamp_mapping.json")
with open(report_path, 'w') as f:
    json.dump(mapping_report, f, indent=2)

print("=" * 90)
print(f"✅ Detailed mapping saved to: {report_path}")
print("=" * 90)
