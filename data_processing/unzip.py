from utils import*

def unzip(params_yaml_path):
    with open(params_yaml_path, 'r') as file:
        params = yaml.safe_load(file)

    zip_file_path = params['base']['zip_file_path']
    output_path = params['unzip']['output_path']

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        # Extract all files and directories from the zip file
        zip_ref.extractall(output_path)
        
        # Move all contents of the first-level subdirectories to the output directory
        for root, dirs, files in os.walk(output_path):
            for dir in dirs:
                # Exclude the output directory itself
                if os.path.join(root, dir) != output_path:
                    # Move contents of the subdirectory to the output directory
                    subdir_path = os.path.join(root, dir)
                    for item in os.listdir(subdir_path):
                        shutil.move(os.path.join(subdir_path, item), output_path)
                    # Remove the now-empty subdirectory
                    os.rmdir(subdir_path)


if __name__ == '__main__':
    unzip('params.yaml')