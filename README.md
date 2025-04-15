# Synthetic Nursing Home Data Generator
*Eva Rombouts - April 2025*
*Version 2.0: A complete rewrite with the knowledge I've gained since creating the first version.*

## Project Overview

This project generates synthetic data for nursing home patient records using various Language Learning Models (LLMs). I aimed to have llms create realistic but artificial client profiles, scenarios, and nursing notes.

## Purpose

The generated synthetic data is primarily intended for:
- Training small language models on nursing home documentation
- Showcasing NLP applications in nursing home settings
- Creating realistic but non-sensitive datasets for healthcare NLP experiments

## Project Structure

The project consists of several Python scripts that execute in sequence:

1. `01save_llm_model_details.py` - Configures and saves LLM model details
2. `02generate_profiles.py` - Generates client profiles using the configured LLMs
3. `03generate_scenarios.py` - Creates scenarios for each client profile 
4. `04generate_records.py` - Generates detailed nursing records based on the scenarios
5. `05combine_data.py` - Combines data from different models into a unified dataset
6. `06category_notes.py` - Generates example notes based on common nursing home topics, not linked to specific client profiles or scenarios.

## Usage

1. Configure LLM credentials in the .env file (.envexample is provided)
2. Check `src\config\llm_config.py` and make adjustments as desired
3. Make sure to add the src folder to your Python path.
4. Run scripts in sequence (01 through 05)
5. Access the generated data in the `/data` directory

## Configuration

The project uses multiple LLM models for data generation:
- OpenAI (in my setting accessed via Azure): GPT-4o-mini and GPT-4o
- Anthropic: Claude-3-5-Sonnet
- Ollama: Phi4

Each model generates data for a specific "ward" in the simulated nursing home, using either somatic or psychogeriatric ward names.

## Technical Components

### LLM Factory
The project uses a factory pattern to manage different LLM providers (OpenAI, Anthropic, Ollama, Azure OpenAI) with a unified interface.

### Response Models
Pydantic models are used to structure the output from LLMs:
- `ClientProfile` - Structure for client profiles
- `ClientScenario` - Structure for weekly scenarios
- `NursesNote` - Structure for nursing records
- `Note` - Structure for categorized notes

### Template Engine
Jinja2 templates are used to construct prompts for the LLM models.

## Requirements

The project requires:
- Python 3.8+
- Pandas for data manipulation
- Jinja2 for templating
- LLM client libraries (OpenAI, Anthropic, etc.)
- Instructor library for structuring LLM outputs with Pydantic
- Access to LLM API endpoints with valid credentials

## Output

The project generates several CSV files:
- `llm_models.csv` - Configuration of LLM models
- `profiles_<model>.csv` - Client profiles for each model
- `scenarios_<model>.csv` - Weekly scenarios for each model
- `records_<model>.csv` - Nursing records for each model
- `profiles.csv`, `scenarios.csv`, `records.csv` - Combined datasets
- `notes.csv` - Categorized nursing notes


## Notes

- The project generates data in Dutch language
- Real patient data is never used; all content is synthetically generated
- The complication library includes realistic events like weight loss, falls, and infections
- Data generation includes randomization for start dates, duration of care, and complications

## License

Please share freely!