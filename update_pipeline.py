from process_directory import extract_and_process_zip
from chunking import chunk_pythoncode_and_add_metadata, chunk_text_and_add_metadata
from embeddings import get_sfr_embedding_model
from gitlab_operations import download_gitlab_repo
from gitlab_operations import get_latest_release_version_tag
from config_loader import ConfigLoader
from packaging.version import Version
from huggingface_operations import upload_folder_to_huggingface, delete_folder_from_huggingface, check_folder_exists
from langchain.vectorstores import Chroma
from datetime import datetime, timedelta, timezone
import time
import tempfile
import json


   
class UpdatePipeline:

    def __init__(self, gitlab_hf_settings, dataset_params_path, update_history_path):
        self.gitlab_hf_settings = gitlab_hf_settings
        self.dataset_params_path = dataset_params_path
        self.update_history_path = update_history_path

        # Preloading configurations for efficiency
        self.gitlab_hf_settings = ConfigLoader(self.gitlab_hf_settings).load()
        self.dataset_params = ConfigLoader(self.dataset_params_path).load()
        self.update_history = ConfigLoader(self.update_history_path).load()


    def is_update_needed(self):
        latest_release_version_tag = self.get_kadiAPY_latest_release_version_tag()
        deployed_release_version_tag = self.get_deployed_vectorstore_version_tag()
        
        is_newer = self.is_newer_version_available(latest_release_version_tag, deployed_release_version_tag)
        
        if is_newer:
            return True, latest_release_version_tag  
        else:
            return False, deployed_release_version_tag
        
    def start_update_pipeline(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Temporary directory created: {temp_dir}")
            
            # Call the download method and unpack the dictionary
            result = self.download_latest_kadiAPY_repo_for_processing(temp_dir)
            target_path_zip_file = result["target_path_zip_file"]
            version_number = result["version_number"]

            kadiAPY_doc_files_content, kadiAPY_doc_files_path = self.get_kadiAPY_doc_dataset(target_path_zip_file)
            print("1:", len(kadiAPY_doc_files_content))
            kadiAPY_library_files_content, kadiAPY_library_files_path = self.get_kadiAPY_library_dataset(target_path_zip_file)
            print("2:", len(kadiAPY_library_files_content))


            kadiAPY_doc_documents = self.chunk_kadiAPY_doc_dataset(kadiAPY_doc_files_content, kadiAPY_doc_files_path)
            print("3:", len(kadiAPY_doc_documents))

            kadiAPY_library_documents = self.chunk_kadiAPY_library_files_dataset(kadiAPY_library_files_content, kadiAPY_library_files_path)
            print("4:", len(kadiAPY_library_documents))

        # temp_dir2 = tempfile.mkdtemp()
        # self.embed_documents_into_vectorstore(kadiAPY_doc_documents + kadiAPY_library_documents, get_sfr_embedding_model(), temp_dir2)
        # self.delete_vectorstore_folder_from_huggingface()
        # self.upload_folder_to_hf(temp_dir2)
        # time.sleep(2)
        # shutil.rmtree(temp_dir, ignore_errors=True)
        # self.update_vectorstore_history(version_number)

    def update_vectorstore_history(self, version_number):
        local_offset_seconds = time.localtime().tm_gmtoff
        local_offset = timedelta(seconds=local_offset_seconds)
        local_time = datetime.now(timezone(local_offset)).isoformat(timespec="seconds")

        new_entry = {
            "update_date": local_time,
            "project_release_version": version_number,
            "deployed_to_hf_repo": self.gitlab_hf_settings["huggingface_parameters"]["hf_repo_id"]
        }

        try:
            with open(self.update_history_path, "r") as file:
                history_data = json.load(file)
        except FileNotFoundError:
            history_data = {"update_history": []}

        history_data["update_history"].insert(0, new_entry)

        with open(self.update_history_path, "w") as file:
            json.dump(history_data, file, indent=2)

        print(f"Updated vectorstore history with version: {version_number}")

    def delete_vectorstore_folder_from_huggingface(self):
        hf_repo_id = self.gitlab_hf_settings["huggingface_parameters"]["hf_repo_id"]
        hf_repo_type = self.gitlab_hf_settings["huggingface_parameters"]["hf_repo_type"]
        hf_vectorstore_path = self.gitlab_hf_settings["huggingface_parameters"]["hf_vectorstore_path"]

        if check_folder_exists(hf_repo_id, hf_vectorstore_path):
            delete_folder_from_huggingface(hf_vectorstore_path, hf_repo_id, hf_repo_type)
        else:
            print(f"Skipping deletion: Folder '{hf_vectorstore_path}' does not exist in repository '{hf_repo_id}'.")

    def upload_folder_to_hf(self, temp_dir):
        hf_repo_id = self.gitlab_hf_settings["huggingface_parameters"]["hf_repo_id"]
        hf_repo_type = self.gitlab_hf_settings["huggingface_parameters"]["hf_repo_type"]
        hf_vectorstore_path = self.gitlab_hf_settings["huggingface_parameters"]["hf_vectorstore_path"]

        upload_folder_to_huggingface(temp_dir, hf_repo_id, hf_repo_type, hf_vectorstore_path)

        upload_folder_to_huggingface(temp_dir, hf_repo_id, hf_repo_type, hf_vectorstore_path)

    def embed_documents_into_vectorstore(self, documents, embedding_model, persist_directory):
        new_vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        new_vectorstore.add_documents(documents)

    def chunk_kadiAPY_doc_dataset(self, doc_files_content, doc_files_path):
        doc_params = self.dataset_params["datasets"]["kadi_apy_docs"]
        chunk_size = doc_params["chunking"]["chunking_size"]
        chunk_overlap = doc_params["chunking"]["chunking_overlap"]
        dataset_name = doc_params["dataset"]
        kadiAPY_doc_documents = chunk_text_and_add_metadata(doc_files_content, doc_files_path, chunk_size, chunk_overlap)
        self.add_dataset_metadata(kadiAPY_doc_documents, dataset_name)
        return kadiAPY_doc_documents

    def chunk_kadiAPY_library_files_dataset(self, kadiAPY_library_files_content, kadiAPY_library_files_content_path):
        library_params = self.dataset_params["datasets"]["kadi_apy_source_code"]
        dataset_name = library_params["dataset"]
        kadiAPY_library_documents = chunk_pythoncode_and_add_metadata(kadiAPY_library_files_content, kadiAPY_library_files_content_path)
        self.add_dataset_metadata(kadiAPY_library_documents, dataset_name)
        return kadiAPY_library_documents
    

    def get_kadiAPY_doc_dataset(self, repo_zip_filepath):
        doc_params = self.dataset_params["datasets"]["kadi_apy_docs"]
        directories_of_kadi_apy_docs = doc_params.get("folder", [])
        return extract_and_process_zip(directories_of_kadi_apy_docs, repo_zip_filepath, filter_filetypes=["rst"])

    def get_kadiAPY_library_dataset(self, repo_zip_filepath):
        library_params = self.dataset_params["datasets"]["kadi_apy_source_code"]
        directory_of_kadi_apy_library_source_code = library_params.get("folder", [])
        return extract_and_process_zip(directory_of_kadi_apy_library_source_code, repo_zip_filepath, filter_filetypes=["py"])

    def add_dataset_metadata(self, documents, dataset_name):
        for doc in documents:
            if hasattr(doc, "metadata"):
                doc.metadata["dataset_category"] = dataset_name
        return documents


    def get_deployed_vectorstore_version_tag(self):
        """Read the used gitlab_project_version from the first entry in the JSON file."""
        data = ConfigLoader(self.update_history_path).load()

        first_entry = data["update_history"][0] 
        return first_entry["project_release_version"]  



    def is_newer_version_available(self, version1, version2):
        """Compare two versions and check if version1 is newer than version2."""
        v1 = Version(version1.lstrip('v'))
        v2 = Version(version2.lstrip('v'))
        
        if v1 > v2:
            return True  # Returns True only if version1 is newer
        else:
            return False  # Returns False if versions are equal or version1 is older

    def get_kadiAPY_latest_release_version_tag(self):
        gitlab_api_url = self.gitlab_hf_settings["gitlab_parameters"]["api_url"]
        gitlab_project_id = self.gitlab_hf_settings["gitlab_parameters"]["project_id"]
   
        return get_latest_release_version_tag(gitlab_api_url, gitlab_project_id)

    def download_latest_kadiAPY_repo_for_processing(self, target_path):
        gitlab_api_url = self.gitlab_hf_settings["gitlab_parameters"]["api_url"]
        gitlab_project_id = self.gitlab_hf_settings["gitlab_parameters"]["project_id"]
        latest_release_version_tag = self.get_kadiAPY_latest_release_version_tag()
        target_path_zip_file = download_gitlab_repo(gitlab_api_url, gitlab_project_id, latest_release_version_tag, target_path)

        return {
            "target_path_zip_file": target_path_zip_file,
            "version_number": latest_release_version_tag
        }
    
    