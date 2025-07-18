import os
import json
from PIL import Image

def create_snippet(img_path, output_path, percentage):
    try:
        img = Image.open(img_path)
        width, height = img.size
        
        new_width = int(width * (percentage / 100))
        new_height = int(height * (percentage / 100))
        
        snippet = img.crop((0, 0, new_width, new_height))
        
        snippet.save(output_path)
        print(f"Create a snippet: {output_path}")
    except Exception as e:
        print(f"Error processing image: {img_path}: {e}")
    
def ensure_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Create directory: {directory}")

def process_dataset(src_dir, percentages):
    images_dir = os.path.join(src_dir, 'images')
    if not os.path.exists(images_dir):
        print(f"Error: Image directory does not exist: {images_dir}")
        return
    
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"Error: No image files in the image directory: {images_dir}")
        return
    
    print(f"Found {len(image_files)} image files")
    
    for percentage in percentages:
        print(f"\nStarting to process {percentage}% segment...")
        snippet_dir = f'./snippet-cub200/snippet-query-{percentage}'
        
        ensure_directory(snippet_dir)
        ensure_directory(os.path.join(snippet_dir, 'images'))
        ensure_directory(os.path.join(snippet_dir, 'passages'))
        ensure_directory(os.path.join(snippet_dir, 'queries'))
        ensure_directory(os.path.join(snippet_dir, 'qrels'))
        
        passages = {}
        for img_file in image_files:
            img_id = str(os.path.splitext(img_file)[0])
            passages[img_id] = {"text": img_file}
        
        with open(os.path.join(snippet_dir, 'passages', 'passages.json'), 'w') as f:
            json.dump(passages, f, indent=4)
        print(f"Create passages.json containing {len(passages)} original image indices")
        
        queries = []
        qrels = []
        
        for img_file in image_files:
            img_id = str(os.path.splitext(img_file)[0])
            base_name, ext = os.path.splitext(img_file)
            snippet_name = f"{base_name}_snippet_{percentage}{ext}"
            
            original_img_path = os.path.join(images_dir, img_file)
            snippet_img_path = os.path.join(snippet_dir, 'images', snippet_name)
            create_snippet(original_img_path, snippet_img_path, percentage)
            
            query_item = {img_id: snippet_name}
            queries.append(query_item)
            
            qrel_item = {img_id: {img_id: 1}}
            qrels.append(qrel_item)
        
        with open(os.path.join(snippet_dir, 'queries', 'queries.json'), 'w') as f:
            json.dump(queries, f, indent=4)
        print(f"Create queries.json containing {len(queries)} processed image segment indices")
        
        with open(os.path.join(snippet_dir, 'qrels', 'qrels.json'), 'w') as f:
            json.dump(qrels, f, indent=4)
        print(f"Create qrels.json containing {len(qrels)} relation items")
        
        print(f"Completed {percentage}% of snippet dataset created in {snippet_dir}")

if __name__ == "__main__":
    percentages = [10, 20, 30, 50, 80]
    
    src_dir = "./eval_exp_visual/processed_beir/StanfordCars/test/snippet_test"
    
    print(f"Start processing the dataset:{src_dir}")
    process_dataset(src_dir, percentages)
    print("All fragment dataset processing completed!")