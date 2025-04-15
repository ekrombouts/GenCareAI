# Generate nurses notes based on a specific category

# This script generates notes for a specific category of care. The categories are chosen based on a study
# conducted in a Dutch nursing home. The notes are generated using different LLM models and are saved to a CSV file.

from pathlib import Path

import pandas as pd
from jinja2 import Environment, FileSystemLoader
from tqdm import tqdm

from llm.llm_factory import LLMFactory
from prompts.category_notes_rm import Note

# --- Configuration ---
datapath = Path(__file__).resolve().parents[1] / "data"
prompts_path = Path(__file__).resolve().parents[1] / "src" / "prompts"

df_models = pd.read_csv(datapath / "llm_models.csv")

num_notes = 50  # Number of notes generated per completion
num_completions = 100  # Number of completions per query


# Load the Jinja2 templates for prompts
env = Environment(loader=FileSystemLoader(prompts_path))
s_template = env.get_template("category_notes_s.jinja")
system_prompt = s_template.render()
u_template = env.get_template("category_notes_u.jinja")


# List of dictionaries containing input data for each category
input_data_list = [
    {
        "cat": "adl",
        "category": "ADL (Algemene Dagelijkse Levensverrichtingen)",
        "examples": """- Dhr. zijn haar gewassen en zijn baard geschoren.
- Inco van mw, was verzadigd vanmorgen en bed was nat.
- Het is niet goed gegaan Mw had een ongelukje met haar kleding en defeaceren Mw was incontinent Mw geholpen met opfrissen en de kleding in de was gedaan
- U bent vanmorgen gedoucht, uw haren zijn gewassen.
""",
        "note_topics": "wassen, aankleden, tanden poetsen, klaarmaken voor de dag, klaarmaken voor de nacht, douchen, gebitsprothese schoonmaken of hulp na incontinentie.",
    },
    {
        "cat": "eten_drinken",
        "category": "eten en drinken",
        "examples": """- Ik kreeg van de dagdienst door dat dhr. zich verslikt in haar drinken. Drinken verdikt aangeboden. Dit ging goed.
- Ochtendzorg verliep goed, dhr was wel zeer vermoeid. Dhr heeft goed gegeten en gedronken. Dhr is na de lunch op bed geholpen om te rusten.
- Nee ik wil niet meer. ik vond niet lekker. Mw heeft ochtend goed gegeten en gedronken. tussen de middageten mw wilde niet. zij heeft paar hapjes vla gegeten en een glas limonade gedronken.
- Mw heeft op bed een paar hapjes pap gegeten.
- De fresubin creme is niet op voorraad. mw ipv de creme fresubin drink aanbieden Fresubin komt vogende week weer binnen.
""",
        "note_topics": "wat de client wel of niet heeft gegeten, welke hulp nodig is bij eten (volledige hulp, aansporing, aangepast bestek of beker), verslikken, bijhouden vocht- en voedingslijst.",
    },
    {
        "cat": "sociaal",
        "category": "sociale interactie en activiteiten",
        "examples": """- Mw. was goed gestemd vanavond en was heel gezellig aanwezig.
- U keek naar de kerkdienst op buurt 4.
- Dhr zit met verschillende medebewoners in de binnentuin.
- Ik eet samen met mijn dochter. We gaan asperges eten.
- Mw. ging haar gangetje. Ging vanmiddag naar een muziek activiteit.
""",
        "note_topics": "georganiseerde activiteiten, het krijgen van bezoek, bladeren door een tijdschriftje, interactie met medebewoners. Hou er rekening mee dat het gaat over rapportages van mensen in een verpleeghuis, met forse beperkingen, dus de sociale interactie en activiteiten zijn beperkt. Meestal betreft het gezelligheid, maar niet altijd.",
    },
    {
        "cat": "huid",
        "category": "huid en wonden",
        "examples": """- ik heb jeuk op mijn rug dhr behandeld met de cetomacrogol creme
- Wat is dat allemaal? Dhr zat aan het verband om zijn arm te plukken. Wondje op arm is klein. Dhr ervaart het verband onprettig. Pleister op het wondje gedaan.
- Dhr zijn liezen zagen er rustig uit. Dhr zijn scrotum ingesmeerd met licht zinkzalf, deze was wel rood. De liezen met beschermende zalf ingesmeerd.
- Mevr. lijkt nu decubitus te ontwikkelen op haar stuit. Mevr. haar hiel verzorgd, dit zag er oke uit, klein beetje geel beslag. Dit schoongemaakt, daarna verbonden volgens plan Dit in de gaten houden.
""",
        "note_topics": "oedeem, decubituswonden, ontvellingen, roodheid en jeuk van de huid. Te lange nagels, smetplekken.",
    },
    {
        "cat": "medisch_logistiek",
        "category": "medische zorg en familie communicatie",
        "examples": """- Oren van mevr zijn uitgespoten er kwam uit beide oren veel viezigheid.
- Graag Dhr morgen wegen
- Arts vragen voor brutans 5 mg besteld
- Dochter van dhr. belde. Ze gaf aan dat ze een aanbod hebben gekregen voor verblijf in een ander verpleeghuis.
- Fam wil graag een gesprek over bezoek cardioloog in het verleden. Er is iets voorgeschreven, ws doorgegeven aan vorige arts. graag contact met familie opnemen voor gesprek of telefonisch gesprek In artsenvisite bespreken
""",
        "note_topics": "zorgplan besprekingen, kleine medische klachten, verzoeken van familie, bestellen van medicijnen.",
    },
    {
        "cat": "nachten",
        "category": "nachten en slapen",
        "examples": """- Mw. heeft de gehele nacht geslapen
- Mw heeft vannacht niet zo goed geslapen. Mw was veel wakker en wat onrustig. Lastig om mw af te leiden en te zorgen dat mw weer wilde slapen. Mw heeft een slechte nachtrust gehad.
- De sensor is de gehele nacht niet afgegaan bij mw
- Dhr. ging rond 23:30 uur naar bed. Heeft de hele nacht geslapen.
- Dhr. was klaarwakker en wilde uit bed en rammelde aan het bedhek. Dhr. vertelde dat hij opgehaald zou worden. Mw. heeft hem overtuigt om toch te gaan slapen en dhr. luisterde naar mw.
""",
        "note_topics": "onrust en dwalen in de nacht, lekker slapen, toiletgang in de nacht, bellen, scheef in bed liggen.",
    },
    {
        "cat": "onrust",
        "category": "onrust, probleemgedrag, stemming",
        "examples": """- Ga opzij. Wat ben jij lelijk Mw schopte naar een andere bewoner en wilde een ander bewoner slaan. Mw een prikkelarme omgeving aangeboden.
- dhr eet de planten van tafel dhr werd begeleid door collega om het uit te spugen werd hier geagiteerd door.
- Waar is het toilet Mag ik al eten Naar zorg toe lopen, zwaaien naar de zorg om hulp.  Mw vraagt veel bevestiging van de zorg,
- Meneer is wat onrustig loopt jammerend heen en weer en zegt steeds erg moe te zijn. Heeft een trieste blik in zijn ogen. Meneer aangeboden om naar bed te gaan, heeft hier geen rust voor.
""",
        "note_topics": "agitatie, onrust, apathie, verwardheid. Meestal is de verwardheid subtiel, maar soms wat heftiger.",
    },
    {
        "cat": "symptomen",
        "category": "ziekte en symptomen",
        "examples": """- Er zat iets vocht in beide voeten. Dhr had vandaag geen steunkousen aan Blijven observeren
- Urine opvangen is tot nu toe nog niet gelukt(mw heeft er steeds def bij) Vanmiddag ook geen pijn gezien alleen evt wat frustratie als iets niet soepel loopt.
- Ik heb pijn dhr gaf pijn aan aan zijn linker pink en ringvinger. Er zitten daar een soort bloedblaren al wel langer. Graag even in de gaten houden en rapporteren of dhr meer pijn krijgt.
- Dhr. had om 6u zeer veel last van slijm en een vieze smaak in zijn mond. Dhr geassisteerd met het spoelen van zijn mond.
- Erg pijnlijk bij de ADL. Morgen graag overleg met de arts over de pijnmedicatie
- Dhr is erg benauwd, klinkt vol, heeft een reutelende ademhaling.
""",
        "note_topics": "pijn, benauwdheid, misselijkheid, diarree, rugklachten, palliatieve zorg. Meestal zijn de klachten subtiel, maar soms heftiger.",
    },
    {
        "cat": "mobiliteit",
        "category": "mobiliteit en transfers",
        "examples": """- Vandaag geholpen met de passieve lift. Dit ging goed.
- Veel rondgelopen vandaag. Mw vergeet steeds haar rollator.
- De banden van de rolstoel zijn zacht. Kan de fysio hier naar kijken?
- De transfers gaan steeds moeilijker. Mw hangt erg in de actieve lift. Glijdt weg. Wil graag nog met de actieve lift geholpen worden, maar dit gaat eigenlijk niet meer. @ Ergo, graag je advies
""",
        "note_topics": "loophulpmiddelen, de rolstoel, valgevaar, valincidenten, transfers, tilliften. De meeste rapportages gaan over dagelijkse dingetjes, dus niet alles is een ernstig incident.",
    },
]

# Initialize list to store records
fn_notes = datapath / f"notes.csv"
notes_list = []

# Loop over llm_models
for _, row_models in tqdm(
    df_models.iterrows(), total=len(df_models), desc="Processing models"
):
    provider = row_models["llm_provider"]
    model = row_models["llm_model"]

    factory = LLMFactory(provider=provider)

    # Loop over input data
    for input_data in input_data_list:
        print(f"Generating notes for {input_data['cat']}")
        user_prompt = u_template.render(
            num_notes=num_notes,
            category=input_data["category"],
            note_topics=input_data["note_topics"],
            examples=input_data["examples"],
        )

        # Generate notes
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        response_model, _ = factory.create_completion(
            response_model=Note,
            messages=messages,
            model=model,
        )

        for note in response_model.note:
            notes_list.append((input_data["cat"], note, model))

    # Create DataFrame from the notes list
    df_notes = pd.DataFrame(notes_list, columns=["category", "note", "model"])

    # Save the DataFrame to a CSV file. The file will be overwritten for each model.
    df_notes.to_csv(fn_notes, index=False)
