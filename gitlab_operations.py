import os
import requests
import urllib.parse
import logging

logging.basicConfig(level=logging.INFO)

import os
import requests
import urllib.parse
import logging

logging.basicConfig(level=logging.INFO)

def download_gitlab_repo(gitlab_api_url, project_id, version, target_dir):
    """
    Downloads the zip file of a released version of a GitLab project.

    Parameters:
        gitlab_api_url (str): The base API URL of the GitLab instance.
        project_id (str): The project ID or path (e.g., 'group/project-name').
        version (str): The release version to download (e.g., 'v1.0.0').
        target_dir (str): The directory where the file will be saved.

    Returns:
        str: The path where the file was saved, or None if the download failed.
    """
    try:
        # Encode the project ID to handle special characters
        encoded_project_id = urllib.parse.quote(project_id, safe="")

        # Construct the URL to download the archive
        url = f"{gitlab_api_url}/projects/{encoded_project_id}/repository/archive.zip?sha={version}"
        logging.info(f"Constructed URL: {url}")

        # Send GET request to download the zip file
        response = requests.get(url, stream=True)
        logging.info(f"HTTP GET Request sent to {url}, Status Code: {response.status_code}")

        if response.status_code == 200:
            # Create the directory if it doesn't exist
            os.makedirs(target_dir, exist_ok=True)

            # Generate the filename dynamically
            filename = f"{project_id.replace('/', '_')}_{version}.zip"  # Example: group_project-name_v1.0.0.zip
            target_path = os.path.join(target_dir, filename)

            # Save the content of the zip file
            with open(target_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)

            logging.info(f"File downloaded successfully and saved as {target_path}")
            return target_path
        else:
            logging.error(f"Failed to download file. HTTP Status Code: {response.status_code}")
            return None

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        return None


import requests

def get_latest_release_version_tag(gitlab_url, project_id):
    """Fetch the latest release version from GitLab's API for public repositories."""

    encoded_project_id = urllib.parse.quote(project_id, safe="")
    url = f'{gitlab_url}/projects/{encoded_project_id}/releases'

    response = requests.get(url)
    
    if response.status_code == 200:
        releases = response.json()
        if releases:
            latest_release = releases[0]  # Assuming releases are ordered by date (latest first)
            latest_version = latest_release['tag_name']
            return latest_version
        else:
            return None
    else:
        print(f"Error fetching releases: {response.status_code}")
        return None




