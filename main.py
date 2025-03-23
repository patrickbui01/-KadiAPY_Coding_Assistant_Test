import logging
from update_pipeline import UpdatePipeline
from config_loader import ConfigLoader


logging.basicConfig(
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def main(): 
    config = ConfigLoader("config_files/gitlab_hf_settings.json").load()
    hf_repo_id = config.get("huggingface_parameters", {}).get("hf_repo_id", "unknown_repo")

    logging.info("Initializing the update pipeline.")
    update_pipeline = UpdatePipeline("config_files/gitlab_hf_settings.json", "config_files/datasets_config.json", "update_history.json")

    logging.info("Checking if an update is needed.")
    is_needed, version_tag = update_pipeline.is_update_needed()

    if is_needed:
        logging.info(f"Update needed. Starting the update pipeline with version tag: {version_tag}")
        update_pipeline.start_update_pipeline()
        logging.info("Update pipeline completed successfully.")
    else:
        logging.info(f"No update needed. The vectorstore in Hugging Face repo '{hf_repo_id}' is up-to-date.")

if __name__ == "__main__":
    logging.info("Starting the script.")
    main()
    logging.info("Script execution finished.")