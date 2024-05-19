import os
from dvclive import Live
import yaml
import json

def compare_datasets(params_yaml_path, live):
    # Checks if the processed data and original data match
    # The function compares the directory structure and the number of images in each directory

    with open(params_yaml_path, 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    
    original_dataset_path = params['evaluate']['ORIGINAL_DATASET_PATH']
    processed_dataset_path = params['evaluate']['PROCESSED_DATASET_PATH']

    # Get the list of directories in the original dataset
    original_dirs = [dir for dir in os.listdir(original_dataset_path) if os.path.isdir(os.path.join(original_dataset_path, dir))]
    
    # Get the list of directories in the processed dataset
    processed_dirs = [dir for dir in os.listdir(processed_dataset_path) if os.path.isdir(os.path.join(processed_dataset_path, dir))]
    
    # Check if the directories match
    if set(original_dirs) != set(processed_dirs):
        # print("Directory structure does not match.")

        return False
    
    # Check the number of images in each directory
    for dir in original_dirs:
        original_num_images = len(os.listdir(os.path.join(original_dataset_path, dir)))
        processed_num_images = len(os.listdir(os.path.join(processed_dataset_path, dir)))
        
        if original_num_images != processed_num_images:
            # print(f"Number of images in {dir} directory does not match: Original {original_num_images}, Processed {processed_num_images}")

            return False

    # print("Dataset structure and number of images match.")

    if not live.summary:
        live.summary = {}
    # Set Matching key as a metric to track in dvclive
    live.summary["Matching"] = True
    return True

if __name__ == '__main__':
    with open('params.yaml', 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    
    output_path = params['evaluate']['OUTPUT_PATH']
    live = Live(os.path.join(output_path,'live'),dvcyaml = False)

    compare_datasets("params.yaml", live)
    live.make_summary()

