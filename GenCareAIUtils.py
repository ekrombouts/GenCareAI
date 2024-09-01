import os

class GenCareAISetup:
    """
    A class to set up the environment for GenCareAI, detecting the environment (Google Colab or Local),
    installing necessary dependencies, and loading API keys.
    """

    def __init__(self):
        """
        Initialize the GenCareAISetup class by detecting the current environment.
        """
        self.environment = self.detect_environment()

    def detect_environment(self):
        """
        Detect whether the script is running in Google Colab or a local environment.

        Returns:
            str: "Google Colab" if running in Colab, otherwise "Local Environment".
        """
        try:
            import google.colab
            return "Google Colab"
        except ImportError:
            return "Local Environment"

    def install_dependencies(self, colab_packages=None, local_packages=None):
        """
        Install necessary dependencies based on the current environment.

        Args:
            colab_packages (list, optional): A list of packages to install if running in Google Colab.
            local_packages (list, optional): A list of packages to install if running in a local environment.

        Example:
            setup.install_dependencies(colab_packages=['langchain', 'chromadb'], local_packages=['plotly', 'dotenv'])
        """
        if self.environment == "Google Colab":
            if colab_packages:
                print("Installing dependencies for Google Colab...")
                for package in colab_packages:
                    !pip install -q {package}
        else:
            if local_packages:
                print("Installing dependencies for Local Environment...")
                for package in local_packages:
                    !pip install -q {package}

    def setup_environment(self, load_openai_keys=True, load_hf_token=True, colab_dir=None, mount_drive=False, openai_key_name='GCI_OPENAI_API_KEY', hf_token_name='HF_TOKEN'):
        """
        Set up the environment by loading necessary API keys and setting the directory.

        Args:
            load_openai_keys (bool, optional): Whether to load the OpenAI API key. Defaults to True.
            load_hf_token (bool, optional): Whether to load the Hugging Face token. Defaults to True.
            colab_dir (str, optional): The directory to change to if running in Colab. Defaults to None.
            mount_drive (bool, optional): Whether to mount Google Drive if running in Colab. Defaults to False.
            openai_key_name (str, optional): The environment variable name for the OpenAI API key. Defaults to 'GCI_OPENAI_API_KEY'.
            hf_token_name (str, optional): The environment variable name for the Hugging Face token. Defaults to 'HF_TOKEN'.

        Returns:
            tuple: Contains the OpenAI API key and Hugging Face token if loaded, otherwise None.

        Example:
            openai_key, hf_token = setup.setup_environment(colab_dir='/content/drive/My Drive/MyProject', mount_drive=True)
        """
        openai_api_key, hf_token = None, None

        if self.environment == "Google Colab":
            from google.colab import userdata
            print("Running in Google Colab")
            if mount_drive:
                from google.colab import drive
                drive.mount('/content/drive')
                if colab_dir:
                    os.chdir(colab_dir)

            if load_openai_keys:
                openai_api_key = userdata.get(openai_key_name)

            if load_hf_token:
                hf_token = userdata.get(hf_token_name)

        else:
            from dotenv import load_dotenv
            print("Running in Local Environment")
            if load_openai_keys or load_hf_token:
                load_dotenv()

            if load_openai_keys:
                openai_api_key = os.getenv(openai_key_name)

            if load_hf_token:
                hf_token = os.getenv(hf_token_name)

        return openai_api_key, hf_token