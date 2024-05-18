from utils import *

def process_image_folder(folder_path):
    data = datasets.ImageFolder(root=folder_path, transform=preprocess)
    return data

def save_transformed_images(dataset, output_folder):
    for path, _ in dataset.samples:
        # Construct source and destination paths
        source_path = path
        destination_path = os.path.join(output_folder, os.path.relpath(source_path, dataset.root))

        # Create destination directory if it doesn't exist
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Copy transformed image to the destination path
        copyfile(source_path, destination_path)


def preprocess(params_yaml_path):

    with open(params_yaml_path, 'r') as file:
        params = yaml.load(file, Loader=yaml.FullLoader)
    
    train_folder = params['preprocess']['train_folder']
    validation_folder = params['preprocess']['valid_folder']
    test_folder = params['preprocess']['test_folder']

    output_folder = params['preprocess']['output_folder']
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
    
    # Recreate the output folder
    os.makedirs(output_folder)

    mean = params['preprocess']['mean']
    std = params['preprocess']['std']

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    folders = [train_folder, validation_folder, test_folder]

    # Load datasets
    try:
        with Pool() as pool:
            data_sets = pool.map(process_image_folder, folders)

        train_data, valid_data, test_data = data_sets

    except Exception as e:
        print(f"Error loading datasets: {e}")
        return
    
    # Save datasets
    output_folder_train = os.path.join(output_folder,"train")
    output_folder_valid = os.path.join(output_folder,"valid")
    output_folder_test = os.path.join(output_folder,"test")


    save_transformed_images(train_data, output_folder_train)
    save_transformed_images(valid_data, output_folder_valid)
    save_transformed_images(test_data, output_folder_test)

if __name__ == '__main__':
    preprocess('params.yaml')

