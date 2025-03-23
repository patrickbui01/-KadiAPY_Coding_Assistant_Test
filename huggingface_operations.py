from huggingface_hub import HfApi, login
import requests

def upload_folder_to_huggingface(folder_path, hf_repo_id, hf_repo_type, vectorstore_path):
    """Uploads a folder to Hugging Face."""
    login()  # Ensure the user is authenticated
    api = HfApi()

    print(f"Uploading folder: {folder_path} to repo: {hf_repo_id} (type: {hf_repo_type})")

    response = api.upload_folder(
        folder_path=folder_path,
        path_in_repo=vectorstore_path,  
        repo_id=hf_repo_id,
        repo_type=hf_repo_type
    )
    

    print("Upload completed with response:", response)
    return response


def delete_folder_from_huggingface(path_in_repo, repo_id, repo_type=None):
    """Deletes a folder from a Hugging Face repository."""

    api = HfApi()

    print(f"Deleting folder: {path_in_repo} from repo: {repo_id} (type: {repo_type})")
    response = api.delete_folder(
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
   
    )
    return response

def check_folder_exists(repo_id, folder_path):
    """
    Checks if a folder exists in a Hugging Face Space repository.

    Parameters:
        repo_id (str): The ID of the Hugging Face Space repository (e.g., "username/repo_name").
        folder_path (str): The folder path to check (relative to the repository root).

    Returns:
        bool: True if the folder exists, False otherwise.
    """
    url = f"https://huggingface.co/spaces/{repo_id}/tree/main/{folder_path}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()  
        return response.status_code == 200
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404: 
            print(f"Error 404: The folder '{folder_path}' does not exist in the repository '{repo_id}'.")
            return False
        else:
            print(f"HTTP error occurred: {http_err}")
            return False
    except requests.exceptions.RequestException as req_err:
        print(f"Request error occurred: {req_err}")
        return False