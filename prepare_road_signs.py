import os
import glob
import xml.etree.ElementTree as ET
from PIL import Image
from sklearn.model_selection import train_test_split
import shutil

ANNOTATIONS_DIR = 'annotations'
IMAGES_DIR = 'images'
OUTPUT_DIR = 'data/RoadSigns'
TRAIN_DIR = os.path.join(OUTPUT_DIR, 'train')
TEST_DIR = os.path.join(OUTPUT_DIR, 'test')
TARGET_SIZE = (28, 28)

def clean_output_dirs():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(TEST_DIR, exist_ok=True)

def parse_and_extract():
    dataset = []
    xml_files = glob.glob(os.path.join(ANNOTATIONS_DIR, '*.xml'))
    
    print(f"Found {len(xml_files)} annotation files.")
    
    for xml_file in xml_files:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        filename = root.find('filename').text
        image_path = os.path.join(IMAGES_DIR, filename)
        
        if not os.path.exists(image_path):
            continue
            
        objects = root.findall('object')
        for i, obj in enumerate(objects):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')
            
            xmin = int(float(bndbox.find('xmin').text))
            ymin = int(float(bndbox.find('ymin').text))
            xmax = int(float(bndbox.find('xmax').text))
            ymax = int(float(bndbox.find('ymax').text))
            
            w = xmax - xmin
            h = ymax - ymin
            if w <= 0 or h <= 0:
                continue
                
            dataset.append({
                'image_path': image_path,
                'class_name': name,
                'box': (xmin, ymin, xmax, ymax)
            })
            
    return dataset

def prepare_dataset():
    clean_output_dirs()
    dataset = parse_and_extract()
    print(f"Extracted {len(dataset)} road signs.")
    
    # Stratified split 80/20
    labels = [d['class_name'] for d in dataset]
    train_data, test_data = train_test_split(dataset, test_size=0.2, stratify=labels, random_state=42)
    
    def process_split(data, split_dir):
        for idx, item in enumerate(data):
            cls_dir = os.path.join(split_dir, item['class_name'])
            os.makedirs(cls_dir, exist_ok=True)
            
            try:
                img = Image.open(item['image_path']).convert('RGB')
                cropped_img = img.crop(item['box'])
                resized_img = cropped_img.resize(TARGET_SIZE, Image.BILINEAR)
                
                # Create unique filename
                base_name = os.path.basename(item['image_path']).split('.')[0]
                save_path = os.path.join(cls_dir, f"{base_name}_{idx}.png")
                resized_img.save(save_path)
            except Exception as e:
                print(f"Error processing {item['image_path']}: {e}")
                
    print(f"Processing {len(train_data)} train images...")
    process_split(train_data, TRAIN_DIR)
    
    print(f"Processing {len(test_data)} test images...")
    process_split(test_data, TEST_DIR)
    
    print("Done! Dataset is ready.")

if __name__ == '__main__':
    prepare_dataset()
