import os
from dotenv import load_dotenv

# Utility functions for GenCareAI notebooks
# Author: Eva Rombouts
# Date: September 1 2024

def check_environment():
    """Determine if the code is running in Google Colab or a local environment."""
    try:
        import google.colab
        return "Google Colab"
    except ImportError:
        return "Local Environment"

def setup_environment(load_openai_keys=True, load_hf_token=True, colab_dir=None, mount_drive=False, openai_key_name='GCI_OPENAI_API_KEY', hf_token_name='HF_TOKEN'):
    """
    Set up the environment by loading necessary API keys and setting the directory.

    Args:
        load_openai_keys (bool): Whether to load the OpenAI API key.
        load_hf_token (bool): Whether to load the Hugging Face token.
        colab_dir (str): The directory to change to if running in Colab. Defaults to None.
        mount_drive (bool): Whether to mount Google Drive if running in Colab.
        openai_key_name (str): The environment variable name for the OpenAI API key.
        hf_token_name (str): The environment variable name for the Hugging Face token.

    Returns:
        tuple: Contains the OpenAI API key and Hugging Face token if loaded, otherwise None.
    """
    env = check_environment()

    if env == "Google Colab":
        print("Running in Google Colab")
        if mount_drive:
            from google.colab import drive
            drive.mount('/content/drive')
            if colab_dir:
                os.chdir(colab_dir)

        if load_openai_keys:
            from google.colab import userdata
            openai_api_key = userdata.get(openai_key_name)
        else:
            openai_api_key = None

        if load_hf_token:
            hf_token = userdata.get(hf_token_name)
        else:
            hf_token = None

    else:
        print("Running in Local Environment")
        openai_api_key, hf_token = None, None
        if load_openai_keys or load_hf_token:
            load_dotenv()
        if load_openai_keys:
            openai_api_key = os.getenv(openai_key_name)
        if load_hf_token:
            hf_token = os.getenv(hf_token_name)

    return openai_api_key, hf_token