# GenCareAI - Data Generation

**Author:** [Eva Rombouts](https://www.doktereva.nl)  
**Date:** 31-08-2024

## Overview

This subdirectory contains a series of notebooks designed to generate, process, and clean synthetic data for a psychogeriatric ward. This data generation is the foundational step for building and fine-tuning models aimed at automating the creation of care plans in nursing homes. 

### Notebook Overview

- __100_GenerateAnonymousCareNotes.ipynb__
    - Purpose: Generates synthetic healthcare notes for NLP experiments using the OpenAI API. These notes are “anonymous,” meaning they are not based on specific client profiles or scenarios, unlike the notes generated in later steps.
    - Input: None directly; uses the GPT-3.5 model for generation; environment variables for API keys and data paths.
    - Output: A CSV file containing categorized synthetic care notes.

- __110_GenerateClientProfiles.ipynb__
    - Purpose: Generates synthetic client profiles for a psychogeriatric ward using the OpenAI GPT-4 model. Profiles include names, types of dementia, physical complaints, and other relevant attributes.
	- Input: None directly; uses internal logic and the GPT-4 model for generation.
	- Output: A CSV file containing the generated client profiles.

- __120_GenerateClientScenarios.ipynb__
	- Purpose: Generates scenarios based on the client profiles created in 110_GenerateClientProfiles.ipynb. Scenarios describe the progression of a client’s condition over several months.
	- Input: Client profiles generated in the previous step.
	- Output: A CSV file containing the generated scenarios.

- __130_RAGIndexing.ipynb__
	- Purpose: Processes the anonymous notes generated earlier, stores them in a Chroma vector database, and enables retrieval of relevant examples for few-shot inference in generating synthetic client records.
	- Input: Anonymous care notes from the previous generation step, stored in Hugging Face datasets; environment variables for API keys and data paths.
	- Output: A populated Chroma vector database that can be queried to retrieve contextually relevant examples for use in generating new care notes.

- __132_VisualiseVectorDB.ipynb__
    - Purpose: Retrieves embeddings and metadata from the Chroma vector database, reduces their dimensionality using t-SNE, and visualizes the results in an interactive Plotly plot.
	- Input: Embeddings and metadata stored in the Chroma vector database (anonymous_notes collection).
	- Output: An interactive t-SNE scatter plot visualizing the relationships between the embedded documents, with hover details for metadata and document content.

- __140_GenerateClientRecords.ipynb__
    - Purpose: Generates synthetic care notes and summaries for clients in a psychogeriatric ward using the OpenAI GPT-3.5-turbo model. The notes are based on the client profiles and scenarios generated above.
	- Input: Client profiles and scenarios generated in earlier steps, Chroma vector database for retrieving example notes, OpenAI GPT-3.5-turbo model for generating care notes and summaries.
	- Output: CSV files containing generated care notes and updated client summaries, structured into comprehensive client records.

- __150_CleanUpNursingHome.ipynb__
    - Purpose: Concatenates and cleans client profiles, scenarios, records, and summaries from different experimental wards to create unified datasets for further analysis.
	- Input: CSV files containing data from multiple wards (Horizon, Venus, Tulip, Cosmos), including client profiles, scenarios, records, and summaries.
	- Output: Four cleaned and structured datasets (Galaxy_clients, Galaxy_scenarios, Galaxy_records, Galaxy_summaries) saved as CSV files and pushed to the Hugging Face Hub.

### How to Run
1.	Environment Setup:
    - Ensure that your environment is properly set up. If using Google Colab, the scripts include installation commands for necessary packages.
    - Store your OpenAI API key in the environment secrets if running in Colab, or in a .env file locally.
2.	Run the Scripts:
    - Execute the notebooks in the order provided above. Each script generates data that is used as input for the next.
3.	Final Data:
    - The final step (ConcatAndCleanData.py) will produce the unified datasets, which are stored in CSV format and can be uploaded to Hugging Face for further use.

### Next Steps
The generated datasets are ready for use in the model-building phase of the GenCare AI project.
