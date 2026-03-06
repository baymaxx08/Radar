import os
import json
from datetime import datetime

data_folder = r"c:\Users\Siddhartha Reddy\Desktop\radar\20250319"

# Define the CSV files and their corresponding data
csv_files_info = {
    '202503019_GH_exposed_corn_ear_cls001.csv': {
        'type': 'Exposed Corn Ear - Close Shot',
        'category': 'exposed',
        'distance': 'close',
        'start_time': '2025-03-19 20:57:55.750',
        'end_time': '2025-03-19 20:58:11.663',
        'records': 391
    },
    '202503019_GH_exposed_corn_ear_far002.csv': {
        'type': 'Exposed Corn Ear - Far Shot',
        'category': 'exposed',
        'distance': 'far',
        'start_time': '2025-03-19 21:01:05.840',
        'end_time': '2025-03-19 21:01:34.023',
        'records': 707
    },
    '202503019_GH_hidden_corn_ear_far004.csv': {
        'type': 'Hidden Corn Ear - Far Shot',
        'category': 'hidden',
        'distance': 'far',
        'start_time': '2025-03-19 21:06:00.316',
        'end_time': '2025-03-19 21:06:23.802',
        'records': 590
    },
    '202503019_GH_hidden_corn_ear_cls005.csv': {
        'type': 'Hidden Corn Ear - Close Shot',
        'category': 'hidden',
        'distance': 'close',
        'start_time': '2025-03-19 21:07:02.931',
        'end_time': '2025-03-19 21:07:21.787',
        'records': 479
    },
    '202503019_GH_stock_cls006.csv': {
        'type': 'Stock (No Cob) - Close Shot',
        'category': 'stock',
        'distance': 'close',
        'start_time': '2025-03-19 21:09:14.198',
        'end_time': '2025-03-19 21:09:33.331',
        'records': 475
    },
    '202503019_GH_stock_far007.csv': {
        'type': 'Stock (No Cob) - Far Shot',
        'category': 'stock',
        'distance': 'far',
        'start_time': '2025-03-19 21:12:15.456',
        'end_time': '2025-03-19 21:12:36.349',
        'records': 552
    }
}

# Image files (in order found)
image_files = ['Image (5).jpg', 'Image (6).jpg', 'Image (7).jpg', 
               'Image (8).jpg', 'Image (9).jpg', 'Image (11).jpg']

# Create comprehensive mapping
mapping = {
    'project': 'Corn Cob Classification System',
    'date': '2025-03-19',
    'image_file_timestamp': '2026-02-19 20:46:18',
    'note': 'Images appear to have been copied/saved on 2026-02-19. EXIF metadata not available. Mapping based on sequential order and CSV file structure.',
    'total_images': len(image_files),
    'total_csv_files': len(csv_files_info),
    'image_to_data_mapping': [],
    'csv_data_summary': csv_files_info,
    'timeline': []
}

# Create sequential mapping (assuming images were taken in order of CSV files)
csv_list = sorted(csv_files_info.keys(), key=lambda x: csv_files_info[x]['start_time'])

for i, (img, csv_file) in enumerate(zip(image_files, csv_list)):
    img_path = os.path.join(data_folder, img)
    csv_info = csv_files_info[csv_file]
    
    mapping['image_to_data_mapping'].append({
        'image': img,
        'image_check': 'Image found' if os.path.exists(img_path) else 'Image NOT FOUND',
        'csv_file': csv_file,
        'data_type': csv_info['type'],
        'category': csv_info['category'],
        'distance': csv_info['distance'],
        'scan_start_time': csv_info['start_time'],
        'scan_end_time': csv_info['end_time'],
        'num_records': csv_info['records'],
        'sequence_number': i + 1
    })

# Create timeline
for csv_file in csv_list:
    csv_info = csv_files_info[csv_file]
    mapping['timeline'].append({
        'time': csv_info['start_time'],
        'file': csv_file,
        'type': csv_info['type'],
        'duration': '~16-19 seconds'
    })

# Save to JSON
output_path = os.path.join(data_folder, 'image_csv_mapping.json')
with open(output_path, 'w') as f:
    json.dump(mapping, f, indent=2)

# Create a readable text report
report_path = os.path.join(data_folder, 'IMAGE_TIMESTAMP_MAPPING.txt')
with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 100 + "\n")
    f.write("PHOTO TO TIMESTAMP MAPPING - CORN COB CLASSIFICATION PROJECT\n")
    f.write("=" * 100 + "\n\n")
    
    f.write(f"Project: {mapping['project']}\n")
    f.write(f"Data Collection Date: {mapping['date']}\n")
    f.write(f"Image File Timestamps: All {mapping['image_file_timestamp']}\n")
    f.write(f"Note: {mapping['note']}\n\n")
    
    f.write("-" * 100 + "\n")
    f.write("TIMELINE OF RADAR SCANS\n")
    f.write("-" * 100 + "\n\n")
    
    for entry in mapping['timeline']:
        f.write(f"📊 {entry['time']}\n")
        f.write(f"   File: {entry['file']}\n")
        f.write(f"   Type: {entry['type']}\n")
        f.write(f"   Duration: ~{entry['duration']}\n\n")
    
    f.write("-" * 100 + "\n")
    f.write("PHOTO TO DATA MAPPING\n")
    f.write("-" * 100 + "\n\n")
    
    for item in mapping['image_to_data_mapping']:
        f.write(f"📷 Image #{item['sequence_number']}: {item['image']}\n")
        f.write(f"   Status: {item['image_check']}\n")
        f.write(f"   Corresponding Data: {item['csv_file']}\n")
        f.write(f"   Data Type: {item['data_type']}\n")
        f.write(f"   Category: {item['category']}\n")
        f.write(f"   Distance: {item['distance']}\n")
        f.write(f"   Scan Time: {item['scan_start_time']} to {item['scan_end_time']}\n")
        f.write(f"   Records: {item['num_records']}\n\n")

# Print results
print("=" * 100)
print("PHOTO TO TIMESTAMP MAPPING - CORN COB CLASSIFICATION PROJECT")
print("=" * 100)
print()
print(f"Project: {mapping['project']}")
print(f"Data Collection Date: {mapping['date']}")
print(f"Image File Timestamps: All {mapping['image_file_timestamp']}")
print()
print(f"Note: {mapping['note']}")
print()

print("-" * 100)
print("TIMELINE OF RADAR SCANS")
print("-" * 100)
print()

for entry in mapping['timeline']:
    print(f"📊 {entry['time']}")
    print(f"   File: {entry['file']}")
    print(f"   Type: {entry['type']}")
    print()

print("-" * 100)
print("PHOTO TO DATA MAPPING")
print("-" * 100)
print()

for item in mapping['image_to_data_mapping']:
    print(f"📷 Image #{item['sequence_number']}: {item['image']}")
    print(f"   Status: {item['image_check']}")
    print(f"   Corresponding Data: {item['csv_file']}")
    print(f"   Data Type: {item['data_type']}")
    print(f"   Scan Time: {item['scan_start_time']} to {item['scan_end_time']}")
    print(f"   Records: {item['num_records']}")
    print()

print("=" * 100)
print(f"✅ JSON mapping saved to: {output_path}")
print(f"✅ Text report saved to: {report_path}")
print("=" * 100)
