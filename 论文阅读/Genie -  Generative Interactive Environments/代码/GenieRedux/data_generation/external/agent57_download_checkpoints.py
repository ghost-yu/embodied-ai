import os
from huggingface_hub import hf_hub_download

def download_model(model_name: str, target_dpath: str = "agent57_checkpoints") -> str:
    # Replace with your repository and file details
    repo_id = "INSAIT-Institute/GenieRedux"  # Your Hugging Face repo ID
    filename = f"{model_name}.ckpt"
    save_dpath = f"{target_dpath}"  # File to download
    #make directories of filename
    os.makedirs(save_dpath, exist_ok=True)
    
    # Download the file
    downloaded_file_path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=save_dpath)
    

    print(f"File downloaded to: {downloaded_file_path}")
    return downloaded_file_path

download_model("Agent57_AdventureIslandII-Nes")
download_model("Agent57_SuperMarioBros-Nes")