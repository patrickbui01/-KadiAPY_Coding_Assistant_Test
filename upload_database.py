from huggingface_hub import HfApi, login
login()
api = HfApi()

try:
    user_info = api.whoami() 
    print(f"Login succeeded! Welcome, {user_info['name']}.")
except Exception as e:
    print(f"Login failed: {e}")


api.upload_folder(
    folder_path="z_no_splitting",
    path_in_repo="data/vectorstore",
    repo_id="bupa1018/KadiAPY_Coding_Assistant",
    repo_type="space",
    ignore_patterns="**/logs/*.txt",
)