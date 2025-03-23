import os
import zipfile
import tempfile
import logging
import traceback

logging.basicConfig(level=logging.INFO)

def process_directory(target_path, base_folder, file_types):
    """
    Recursively collects files of the specified types from the given directory, 
    prepending the base folder to the file paths, and retrieves their content.

    Parameters:
        target_path (str): The path of the directory to process.
        base_folder (str): The base folder to prepend to the paths.
        file_types (list): A list of file types to include (e.g., ["python", "rst"]).

    Returns:
        list: A list of tuples containing file paths and their content.
    """
    files = []
    valid_extensions = { 
        "py": ".py",
        "rst": ".rst"
    }
    

    if file_types is None:
        filetypes_to_include_in_processing = list(valid_extensions.values())
    else:
        filetypes_to_include_in_processing = [valid_extensions[file_type] for file_type in file_types if file_type in valid_extensions]
                                 

    for root, _, filenames in os.walk(target_path):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in filetypes_to_include_in_processing):
                # Generate the full path and strip everything before the base folder
                full_path = os.path.join(root, filename)
                stripped_path = os.path.relpath(full_path, target_path)  # Keep file path relative to `target_path`
                modified_path = os.path.join(base_folder, stripped_path).replace("\\", "/") 
                # Read file content
                try:
                    with open(full_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                    files.append((modified_path, content))
                except Exception as e:
                    logging.error(f"Failed to read file: {full_path}, Error: {e}")
                    files.append((modified_path, None)) 
    return files


def extract_and_process_zip(folder_paths, zip_path, filter_filetypes = None):
    """
    Unzips a file to a temporary directory, walks through subdirectories, and matches specified folder paths.
    Processes files of the specified types.

    Parameters:
        folder_paths (list): A list of subdirectory paths to match within the unzipped content.
        zip_path (str): The path to the zip file to process.
        filter_filetypes (list): A list of file types to include (e.g., ["python", "rst"]).

    Returns:
        list: A list of tuples containing modified paths and their content for matched files.
    """
    matched_paths = []
    matched_contents = []
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)

            root_dir_name = next(os.walk(temp_dir))[1][0]
            root_dir_path = os.path.join(temp_dir, root_dir_name)

            for root, dirs, _ in os.walk(root_dir_path):
                relative_path = os.path.relpath(root, root_dir_path).replace("\\", "/")

                for folder_path in folder_paths:
                    if folder_path.lstrip("/") in relative_path:

                        base_folder = folder_path.lstrip("/")
                        files = process_directory(root, base_folder, filter_filetypes)

                        for path, content in files:
                            matched_contents.append(content)
                            matched_paths.append(path)

                        dirs.clear()
                        break

    except FileNotFoundError as e:
        logging.error(f"ZIP file not found: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        logging.error(traceback.format_exc())

    return matched_contents, matched_paths