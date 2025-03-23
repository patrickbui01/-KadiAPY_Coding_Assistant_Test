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
from datetime import datetime, timedelta, timezone
import tempfile
import json

class UpdatePipeline:
    def __init__(self):
        # Initialize the pipeline with a store name and data to update
        pass

    def is_update_needed(self, gitlab_config_path, deployed_vectorstore_config_path):
        latest_release_version_tag = self.get_kadiAPY_latest_release_version_tag(gitlab_config_path)
        deployed_release_version_tag = self.get_deployed_vectorstore_version_tag(deployed_vectorstore_config_path)

        if self.is_newer_version_available(latest_release_version_tag, deployed_release_version_tag) == latest_release_version_tag:
            return True, latest_release_version_tag
        else:
            return False, deployed_release_version_tag


    def start_update_pipeline(self):
       
        huggingface_config_path = "huggingface_parameters.json"
        gitlab_config_path = "gitlab_parameters.json"
        dataset_kadi_apy_doc_params_path = "dataset_kadi_apy_doc_params.json"
        dataset_kadi_apy_library_params_path = "dataset_kadi_apy_library_params.json"
        update_history_path = "update_history.json"

        with tempfile.TemporaryDirectory() as temp_dir:
            print(f"Temporary directory created: {temp_dir}")
            
            # Call the download method and unpack the dictionary
            result = self.download_latest_kadiAPY_repo_for_processing(gitlab_config_path, temp_dir)
            target_path_zip_file = result["target_path_zip_file"] 
            version_number = result["version_number"]            

            # Pass the ZIP file path to the next methods
            kadiAPY_doc_files_content, kadiAPY_doc_files_path = self.get_kadiAPY_doc_dataset(dataset_kadi_apy_doc_params_path, target_path_zip_file)
            kadiAPY_library_files_content, kadiAPY_library_files_path = self.get_kadiAPY_library_dataset(dataset_kadi_apy_library_params_path, target_path_zip_file)


        kadiAPY_doc_documents = self.chunk_kadiAPY_doc_dataset(kadiAPY_doc_files_content, kadiAPY_doc_files_path, dataset_kadi_apy_doc_params_path )
        kadiAPY_library_documents = self.chunk_kadiAPY_library_files_dataset(kadiAPY_library_files_content, kadiAPY_library_files_path, dataset_kadi_apy_library_params_path )

        with tempfile.TemporaryDirectory() as temp_dir2:
            self.embed_documents_into_vectorstore(kadiAPY_doc_documents + kadiAPY_library_documents, get_sfr_embedding_model(), temp_dir)
            self.delete_vectorstore_folder_from_huggingface(huggingface_config_path)
            self.upload_folder_to_hf(huggingface_config_path, temp_dir)
        
        self.update_vectorstore_history(version_number, update_history_path, huggingface_config_path)





    def update_vectorstore_history(self, version_number, update_history_file_path, hugginface_config_file):

        config_loader = ConfigLoader(update_history_file_path)
        history_data = config_loader.load()
        
        config_loader = ConfigLoader(hugginface_config_file)
        huggingface_params = config_loader.load()
        target_hf_repo_id = huggingface_params["hf_repo_id"]      

        local_offset_seconds = time.localtime().tm_gmtoff
        local_offset = timedelta(seconds=local_offset_seconds)
        local_time = datetime.now(timezone(local_offset)).isoformat(timespec="seconds")
        
        new_entry = {
            "update_date": local_time,
            "project_release_version": version_number,
            "deployed_to_hf_repo": target_hf_repo_id
        }
        try:
            with open(update_history_file_path, "r") as file:
                history_data = json.load(file)
        except FileNotFoundError:
            history_data = {"update_history": []}

        # Insert the new entry at the beginning of the list
        history_data["update_history"].insert(0, new_entry)

        with open(update_history_file_path, "w") as file:
            json.dump(history_data, file, indent=2)

        print(f"Updated vectorstore history with version: {version_number}")



    def delete_vectorstore_folder_from_huggingface(self, huggingface_config_path):
        if not huggingface_config_path:
            print("Hugging Face config path is invalid or missing.")
            return
        
        config_loader = ConfigLoader(huggingface_config_path)
        config_data = config_loader.load()

        hf_repo_id = config_data["hf_repo_id"]
        hf_repo_type = config_data["hf_repo_type"]
        hf_vectorstore_path = config_data["hf_vectorstore_path"]
        
        if check_folder_exists(hf_repo_id, hf_vectorstore_path):
            delete_folder_from_huggingface(hf_vectorstore_path, hf_repo_id, hf_repo_type)
        else:
            print(f"Skipping deletion: Folder '{hf_vectorstore_path}' does not exist in repository '{hf_repo_id}'.")


    

    def upload_folder_to_hf(self, huggingface_config_path, temp_dir):
        
        config_loader = ConfigLoader(huggingface_config_path)
        config_data = config_loader.load()
        hf_repo_id = config_data["hf_repo_id"]
        hf_repo_type =config_data["hf_repo_type"]
        hf_vectorstore_path = config_data["hf_vectorstore_path"]

        upload_folder_to_huggingface(temp_dir,hf_repo_id, hf_repo_type, hf_vectorstore_path )
    
    def embed_documents_into_vectorstore(self, documents, embedding_model, persist_directory):
        new_vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding_model)
        new_vectorstore.add_documents(documents)


    def chunk_kadiAPY_doc_dataset(self, doc_files_content, doc_files_path, dataset_kadi_apy_doc_params_path):
        config_loader = ConfigLoader(dataset_kadi_apy_doc_params_path)
        print(config_loader)
        config_data = config_loader.load()


        chunk_size = config_data["chunking"]["chunking_size"]
        chunk_overlap = config_data["chunking"]["chunking_overlap"]
        dataset_name = config_data["dataset"]
        kadiAPY_doc_documents = chunk_text_and_add_metadata(doc_files_content, doc_files_path, chunk_size, chunk_overlap)
        print(len(kadiAPY_doc_documents))
        self.add_dataset_metadata(kadiAPY_doc_documents, dataset_name)

        return kadiAPY_doc_documents


    def chunk_kadiAPY_library_files_dataset(self,kadiAPY_library_files_content, kadiAPY_library_files_content_path, dataset_kadi_apy_library_params):
        config_loader = ConfigLoader(dataset_kadi_apy_library_params)
        config_data = config_loader.load()
        dataset_name = config_data["dataset"]
        kadiAPY_library_documents= chunk_pythoncode_and_add_metadata(kadiAPY_library_files_content, kadiAPY_library_files_content_path)
        self.add_dataset_metadata(kadiAPY_library_documents, dataset_name)
        return kadiAPY_library_documents


    def get_kadiAPY_doc_dataset(self, dataset_kadi_apy_doc_params_path, repo_zip_filepath, filter_filetype = ["rst"]):
        # Load configuration
        config_loader = ConfigLoader(dataset_kadi_apy_doc_params_path)
        config_data = config_loader.load()

        directories_of_kadi_apy_docs = config_data.get("folder", [])
        return extract_and_process_zip(directories_of_kadi_apy_docs, repo_zip_filepath, filter_filetype)
    

    def add_dataset_metadata(self, documents, dataset_name):
        dataset_category = dataset_name  
        
        for doc in documents:  
            if hasattr(doc, "metadata"): 
                doc.metadata["dataset_category"] = dataset_category 
        
        return documents 

    def get_kadiAPY_library_dataset(self, dataset_kadi_apy_doc_params_path, target_path, filter_filetype = ["py"]):
        # Load configuration
        config_loader = ConfigLoader(dataset_kadi_apy_doc_params_path)
        config_data = config_loader.load()

        directory_of_kadi_apy_library_source_code = config_data.get("folder", [])
        return extract_and_process_zip(directory_of_kadi_apy_library_source_code, target_path, filter_filetype)


    def get_deployed_vectorstore_version_tag(self, deployed_vectorstore_history_json_path): 
        """read the used gitlab_project_version from the first entry in the JSON file."""
        config_loader = ConfigLoader(deployed_vectorstore_history_json_path)
        data = config_loader.load()

        first_entry = data["deployed_vectorstore_history"][0]
        return self.first_entry["project_release_version"]

    def get_kadiAPY_latest_release_version_tag(self, gitlab_parameter_config_path):
        gitlab_config_loader = ConfigLoader(gitlab_parameter_config_path)
        gitlab_config = gitlab_config_loader.load()
        gitlab_api_url = gitlab_config.get("api_url")
        gitlab_project_id = gitlab_config.get("project id")     
        return get_latest_release_version_tag(gitlab_api_url, gitlab_project_id)


    def is_newer_version_available(version1, version2):
        """Compare two versions and check if version1 is newer than version2."""
        v1 = Version(version1.lstrip('v'))  
        v2 = Version(version2.lstrip('v'))

        if v1 > v2:
            return version1  
        else:
            return version2  


    def download_latest_kadiAPY_repo_for_processing(self, gitlab_parameter_config_path, target_path):
        gitlab_config_loader = ConfigLoader(gitlab_parameter_config_path)
        gitlab_config = gitlab_config_loader.load()
        gitlab_api_url = gitlab_config.get("api_url")
        gitlab_project_id = gitlab_config.get("project id")
        latest_release_version_tag = self.get_kadiAPY_latest_release_version_tag(gitlab_parameter_config_path)
        target_path_zip_file = download_gitlab_repo(gitlab_api_url, gitlab_project_id, latest_release_version_tag, target_path)

        return {
            "target_path_zip_file": target_path_zip_file,
            "version_number": latest_release_version_tag
        }
        

    