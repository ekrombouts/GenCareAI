import os
import random
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from llm.llm_factory import LLMFactory

datapath = Path(__file__).resolve().parents[1] / "data"

df_llm_metadata = pd.read_csv(datapath / "llm_metadata.csv")
df_llm_metadata = df_llm_metadata.iloc[[3]]

temp = 0.9
duration = 20
duration_sd = 6
num_complications_min = 1
num_complications_max = 3


complications_library = [
    "gewichtsverlies",
    "algehele achteruitgang",
    "decubitus",
    "urineweginfectie",
    "pneumonie",
    "delier",
    "verergering van onderliggende lichamelijke klachten",
    "verbetering van de klachten",
    "overlijden",
    "valpartij",
]


class ClientScenario(BaseModel):
    week: int = Field(description="Weeknummer")
    events_description: str = Field(
        description="Beschrijving van de gebeurtenissen en zorg"
    )


class ClientScenarios(BaseModel):
    scenario: List[ClientScenario]


def determine_duration(mean=12, std_dev=4):
    return int(np.round(np.random.normal(mean, std_dev)))


def sample_complications(complications_library, min_n=1, max_n=3):
    num = random.randint(min_n, max_n)
    return ", ".join(random.sample(complications_library, num))


def format_client_profile(row: pd.Series) -> str:
    return (
        f"Naam: {row['naam']}\n"
        f"Diagnose: {row['diagnose']}\n"
        f"Lichamelijke klachten: {row['somatiek']}\n"
        f"ADL: {row['adl']}\n"
        f"Mobiliteit: {row['mobiliteit']}\n"
        f"Gedrag: {row['gedrag']}"
    )


system_prompt = (
    "Je bent een behulpzame assistent, die synthetische zorgscenario's genereert."
)

prompt_template = """
Dit is het profiel van een fictieve client in het verpleeghuis:
---
{client_profile}
---

Maak een week-tot-week tijdlijn voor een periode van {num_weeks} weken, en beschrijf hierin beloop van zijn/haar verblijf in het verpleeghuis.
Verwerk de volgende complicatie(s) hierin: {complications}.

Instructies:
- Formuleer elke scenarioregel zodanig dat deze duidelijk en begrijpelijk is voor een taalmodel.
- Beperk dramatische veranderingen en focus op subtiele ontwikkelingen.
- Vermijd het noemen van de naam van de client.
"""

for _, llm_details in df_llm_metadata.iterrows():
    llm_provider = llm_details["llm_provider"]
    model_name = llm_details["llm_model"]

    fn_profiles = datapath / f"client_profiles_{model_name}.csv"
    df = pd.read_csv(fn_profiles)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fn_scenarios = datapath / f"client_scenarios_{model_name}_{timestamp}.csv"

    if not os.path.exists(fn_scenarios):
        scenario_list = []
        factory = LLMFactory(provider=llm_provider)

        for _, row in tqdm(
            df.iterrows(),
            total=df.shape[0],
            desc=f"Generating Scenario's for {model_name}",
        ):
            profile = format_client_profile(row)
            num_weeks = determine_duration(mean=duration, std_dev=duration_sd)
            complications = sample_complications(
                complications_library, num_complications_min, num_complications_max
            )

            user_prompt = prompt_template.format(
                client_profile=profile, num_weeks=num_weeks, complications=complications
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            response_model, _ = factory.create_completion(
                response_model=ClientScenarios,
                messages=messages,
                model=model_name,
                temperature=temp,
            )

            for scenario in response_model.scenario:
                scenario_list.append(
                    (
                        row["client_id"],
                        scenario.week,
                        scenario.events_description,
                        complications,
                        num_weeks,
                    )
                )

        df_scenarios = pd.DataFrame(
            scenario_list,
            columns=[
                "client_id",
                "week",
                "events_description",
                "complications",
                "num_weeks",
            ],
        )
        df_scenarios.insert(0, "scenario_id", range(1, len(df_scenarios) + 1))
        df_scenarios.to_csv(fn_scenarios, index=False)
        print(f"Data saved to {fn_scenarios}.")
    else:
        print("Scenario file found. Loading data...")
        df_scenarios = pd.read_csv(fn_scenarios)
