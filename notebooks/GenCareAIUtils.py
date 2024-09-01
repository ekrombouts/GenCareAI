import os

class GenCareAISetup:
    """
    A class to set up the environment for GenCareAI, detecting the environment (Google Colab or Local),
    and providing methods to retrieve API keys and handle file paths.
    """

    def __init__(self):
        """
        Initialize the GenCareAISetup class by detecting the current environment,
        attempting to mount Google Drive if in Colab, and setting the base directory.
        """
        self.environment = self.detect_environment()
        self.base_dir = self.set_base_dir()

    def detect_environment(self):
        """
        Detect whether the script is running in Google Colab or a local environment.

        Returns:
            str: "Colab" if running in Colab, otherwise "Local".
        """
        try:
            import google.colab
            return "Colab"
        except ImportError:
            return "Local"

    def mount_drive_if_colab(self):
        """
        Attempt to mount Google Drive if running in Colab, without raising an error if it fails.
        """
        if self.environment == "Colab":
            try:
                from google.colab import drive
                drive.mount('/content/drive')
                print("Google Drive mounted.")
            except Exception as e:
                print("Google Drive could not be mounted. Continuing without drive access.")

    def set_base_dir(self):
        """
        Set the base directory depending on the environment.

        Returns:
            str: The base directory path to be used for file operations.
        """
        if self.environment == "Colab":
            self.mount_drive_if_colab()
            return '/content/drive/My Drive/Colab Notebooks/GenCareAI/scripts'
        else:
            return os.getcwd()

    def get_file_path(self, relative_path):
        """
        Get the full file path based on the environment and the base directory.

        Args:
            relative_path (str): The relative path of the file from the base directory.

        Returns:
            str: The full file path.
        """
        return os.path.join(self.base_dir, relative_path)

    def get_openai_key(self, key_name='GCI_OPENAI_API_KEY'):
        """
        Retrieve the OpenAI API key from the environment.

        Args:
            key_name (str, optional): The environment variable name for the OpenAI API key. Defaults to 'GCI_OPENAI_API_KEY'.

        Returns:
            str: The OpenAI API key, or None if not found.
        """
        if self.environment == "Colab":
            from google.colab import userdata
            return userdata.get(key_name)
        else:
            from dotenv import load_dotenv
            load_dotenv()
            return os.getenv(key_name)

    def get_hf_token(self, token_name='HF_TOKEN'):
        """
        Retrieve the Hugging Face token from the environment.

        Args:
            token_name (str, optional): The environment variable name for the Hugging Face token. Defaults to 'HF_TOKEN'.

        Returns:
            str: The Hugging Face token, or None if not found.
        """
        if self.environment == "Colab":
            from google.colab import userdata
            return userdata.get(token_name)
        else:
            from dotenv import load_dotenv
            load_dotenv()
            return os.getenv(token_name)