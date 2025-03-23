from update_pipeline import UpdatePipeline


update = UpdatePipeline()
# #update.delete_vectorstore_folder_from_huggingface("huggingface_parameters.json")
update.start_update_pipeline()
