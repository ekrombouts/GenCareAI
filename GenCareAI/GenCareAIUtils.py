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
        Attempt to mount Google Drive if running in Colab.
        """
        if self.environment == "Colab":
            try:
                from google.colab import drive

                drive.mount("/content/drive")
                print("Google Drive mounted.")
            except Exception as e:
                print(
                    "Google Drive could not be mounted. Continuing without drive access."
                )

    def set_base_dir(
        self,
        colab_dir="/content/drive/My Drive/Colab Notebooks/GenCareAI",
        local_dir="GenCareAI",
    ):
        """
        Set the base directory depending on the environment.

        Args:
            colab_dir (str, optional): The base directory path to be used in Colab. Defaults to a typical Colab-specific path.
            local_dir (str, optional): The base directory path to be used locally. Defaults to 'GenCareAI'.

        Returns:
            str: The base directory path to be used for file operations.
        """
        if self.environment == "Colab":
            self.mount_drive_if_colab()
            return colab_dir
        else:
            current_path = os.getcwd()
            if local_dir in current_path:
                # Assuming the base_dir is a parent folder that contains "local_dir"
                base_dir = current_path.split(local_dir)[0] + local_dir
                return base_dir
            else:
                # If not found, assume the current directory is the base_dir
                return current_path

    def get_file_path(self, relative_path):
        """
        Get the full file path based on the environment and the base directory.

        Args:
            relative_path (str): The relative path of the file from the base directory.

        Returns:
            str: The full file path.
        """
        return os.path.join(self.base_dir, relative_path)

    def get_openai_key(self, key_name="GCI_OPENAI_API_KEY"):
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

    def get_hf_token(self, token_name="HF_TOKEN"):
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


class ClientProfileFormatter:
    """
    A class to format client profiles and determine the gender of a client based on their name.
    """

    def format_client_profile(
        self, profile_row, display_name=True, display_diagnosis=True
    ):
        """
        Format profile information from a given profile_row.

        Args:
            profile_row (pd.Series): A Pandas Series object containing the client's profile information.
            display_name (bool, optional): Whether to display the client's name. Defaults to True.
            display_diagnosis (bool, optional): Whether to display the diagnosis. Defaults to True.

        Returns:
            str: A formatted profile as a string.
        """
        profile = ""
        if display_name:
            profile += f"Naam: {profile_row['naam']}\n"
        else:
            gender = self.determine_client_gender(profile_row)
            if gender == "female":
                profile += "Geslacht: Vrouw\n"
            elif gender == "male":
                profile += "Geslacht: Man\n"
            else:
                profile += "Geslacht: Onbekend\n"

        if display_diagnosis:
            profile += f"Type Dementie: {profile_row['type_dementie']}\n"
        profile += (
            f"Lichamelijke klachten: {profile_row['somatiek']}\n"
            f"ADL: {profile_row['adl']}\n"
            f"Mobiliteit: {profile_row['mobiliteit']}\n"
            f"Cognitie/gedrag: {profile_row['gedrag']}"
        )

        return profile

    def determine_client_gender(self, profile_row):
        """
        Determines the gender of the client based on their name.

        Args:
            profile_row (pd.Series): A Pandas Series object containing the client's profile information.

        Returns:
            str: 'female' if the name contains 'Mevrouw', 'male' if it contains 'Meneer',
                 and 'unknown' if neither is found.
        """
        name = profile_row["naam"]
        if "Mevrouw" in name:
            return "female"
        elif "Meneer" in name:
            return "male"
        else:
            return "unknown"
