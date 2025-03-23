from update_pipeline import UpdatePipeline


# Assuming you have already created an instance of UpdatePipeline
update_pipeline = UpdatePipeline("gitlab_parameters.json", "huggingface_parameters.json", "dataset_kadi_apy_doc_params.json", "dataset_kadi_apy_library_params.json", "update_history.json")

# Call the method and unpack the returned values
is_needed, version_tag = update_pipeline.is_update_needed()
print(is_needed)
if is_needed:
    update_pipeline.start_update_pipeline()
