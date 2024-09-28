"""
Author: Eva Rombouts
Date: 2024-09-23

Description:
"""

from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List

# Constants & variables
num_completions = 2
llm_model = "gpt-4o-mini-2024-07-18"


class Note(BaseModel):
    note: List[str] = Field(description="rapportage tekst")


output_parser = PydanticOutputParser(pydantic_object=Note)
format_instructions = output_parser.get_format_instructions()

template = """
Dit zijn voorbeelden van rapportages over {category}:

{examples}

Andere rapportages kunnen bijvoorbeeld gaan over: {note_topics}

Verzin 10 van zulke rapportages. Varieer met de zinsopbouw en stijl.

{format_instructions}
"""

prompt_template = PromptTemplate(
    input_variables=["category", "examples", "note_topics"],
    template=template,
    partial_variables={"format_instructions": format_instructions},
)

llm = ChatOpenAI(model=llm_model, temperature=1.1, n=num_completions)

input_data_list = [
    {
        "category": "ADL (Activiteiten van het Dagelijks Leven)",
        "examples": """- Dhr. zijn haar gewassen en zijn baard geschoren.
- Inco van mw, was verzadigd vanmorgen en bed was nat.
- Het is niet goed gegaan Mw had een ongelukje met haar kleding en defeceren. Mw was incontinent. Mw geholpen met opfrissen en de kleding in de was gedaan.
- U bent vanmorgen gedoucht, uw haren zijn gewassen.
""",
        "note_topics": "wassen, aankleden, tanden poetsen, klaarmaken voor de dag, klaarmaken voor de nacht, douchen, gebitsprothese schoonmaken of hulp na incontinentie.",
    },
    {
        "category": "Eten en drinken",
        "examples": """- Ik kreeg van de dagdienst door dat dhr. zich verslikt in haar drinken. Drinken verdikt aangeboden. Dit ging goed.
- Ochtendzorg verliep goed, dhr was wel zeer vermoeid. Dhr heeft goed gegeten en gedronken. Dhr is na de lunch op bed geholpen om te rusten.
- Nee ik wil niet meer. Ik vond het niet lekker. Mw heeft 's ochtends goed gegeten en gedronken. Tussen de middag wilde mw niet eten. Zij heeft paar hapjes vla gegeten en een glas limonade gedronken.
- Mw heeft op bed een paar hapjes pap gegeten.
- De Fresubin crème is niet op voorraad. Mw ipv de crème Fresubin drink aanbieden. Fresubin komt volgende week weer binnen.
""",
        "note_topics": "wat de cliënt wel of niet heeft gegeten, welke hulp nodig is bij eten (volledige hulp, aansporing, aangepast bestek of beker), verslikken, bijhouden vocht- en voedingslijst.",
    },
    {
        "category": "Sociaal",
        "examples": """- Mw. was goed gestemd vanavond en was heel gezellig aanwezig.
- U keek naar de kerkdienst op buurt 4.
- Dhr zit met verschillende medebewoners in de binnentuin.
- Ik eet samen met mijn dochter. We gaan asperges eten.
- Mw. ging haar gangetje. Ging vanmiddag naar een muziekactiviteit.
""",
        "note_topics": "georganiseerde activiteiten, het krijgen van bezoek, bladeren door een tijdschriftje, interactie met medebewoners.",
    },
    {
        "category": "Huid en wonden",
        "examples": """- Ik heb jeuk op mijn rug. Dhr behandeld met de cetomacrogol crème.
- "Wat is dat allemaal?" Dhr zat aan het verband om zijn arm te plukken. Wondje op arm is klein. Dhr ervaart het verband onprettig. Pleister op het wondje gedaan.
- Dhr zijn liezen zagen er rustig uit. Dhr zijn scrotum ingesmeerd met licht zinkzalf; deze was wel rood. De liezen met beschermende zalf ingesmeerd.
- Mevr. lijkt nu decubitus te ontwikkelen op haar stuit. Mevr. haar hiel verzorgd, dit zag er oké uit, klein beetje geel beslag. Dit schoongemaakt, daarna verbonden volgens plan. Dit in de gaten houden.
""",
        "note_topics": "oedeem, decubituswonden, ontvellingen, roodheid en jeuk van de huid, te lange nagels, smetplekken.",
    },
    {
        "category": "Medisch logistiek",
        "examples": """- Oren van mevr zijn uitgespoten; er kwam uit beide oren veel viezigheid.
- Graag dhr morgen wegen.
- Arts vragen voor Brutasal 5 mg besteld.
- Dochter van dhr. belde. Ze gaf aan dat ze een aanbod hebben gekregen voor verblijf in een ander verpleeghuis.
- Familie wil graag een gesprek over bezoek cardioloog in het verleden. Er is iets voorgeschreven, waarschijnlijk doorgegeven aan vorige arts. Graag contact met familie opnemen voor gesprek of telefonisch gesprek. In artsenvisite bespreken.
""",
        "note_topics": "zorgplanbesprekingen, kleine medische klachten, verzoeken van familie, bestellen van medicijnen.",
    },
    {
        "category": "Nachten en slapen",
        "examples": """- Mw. heeft de gehele nacht geslapen.
- Mw heeft vannacht niet zo goed geslapen. Mw was veel wakker en wat onrustig. Lastig om mw af te leiden en te zorgen dat mw weer wilde slapen. Mw heeft een slechte nachtrust gehad.
- De sensor is de gehele nacht niet afgegaan bij mw.
- Dhr. ging rond 23:30 uur naar bed. Heeft de hele nacht geslapen.
- Dhr. was klaarwakker en wilde uit bed en rammelde aan het bedhek. Dhr. vertelde dat hij opgehaald zou worden. Mw. heeft hem overtuigd om toch te gaan slapen en dhr. luisterde naar mw.
""",
        "note_topics": "onrust en dwalen in de nacht, lekker slapen, toiletgang in de nacht, bellen, scheef in bed liggen.",
    },
    {
        "category": "Onrust en gedrag",
        "examples": """- "Ga opzij. Wat ben jij lelijk." Mw schopte naar een andere bewoner en wilde een andere bewoner slaan. Mw een prikkelarme omgeving aangeboden.
- Dhr eet de planten van tafel. Dhr werd begeleid door collega om het uit te spugen; werd hier geagiteerd door.
- "Waar is het toilet? Mag ik al eten?" Naar zorg toe lopen, zwaaien naar de zorg om hulp. Mw vraagt veel bevestiging van de zorg.
- Meneer is wat onrustig, loopt jammerend heen en weer en zegt steeds erg moe te zijn. Heeft een trieste blik in zijn ogen. Meneer aangeboden om naar bed te gaan, heeft hier geen rust voor.
""",
        "note_topics": "agitatie, onrust, apathie, verwardheid. Meestal is de verwardheid subtiel, maar soms wat heftiger.",
    },
    {
        "category": "Symptomen van ziekte",
        "examples": """- Er zat iets vocht in beide voeten. Dhr had vandaag geen steunkousen aan. Blijven observeren.
- Urine opvangen is tot nu toe nog niet gelukt (mw heeft er steeds def bij). Vanmiddag ook geen pijn gezien, alleen eventueel wat frustratie als iets niet soepel loopt.
- "Ik heb pijn." Dhr gaf pijn aan aan zijn linker pink en ringvinger. Er zitten daar een soort bloedblaren, al wel langer. Graag even in de gaten houden en rapporteren of dhr meer pijn krijgt.
- Dhr. had om 6 uur zeer veel last van slijm en een vieze smaak in zijn mond. Dhr geassisteerd met het spoelen van zijn mond.
- Erg pijnlijk bij de ADL. Morgen graag overleg met de arts over de pijnmedicatie.
- Dhr is erg benauwd, klinkt vol, heeft een reutelende ademhaling.
""",
        "note_topics": "pijn, benauwdheid, misselijkheid, diarree, rugklachten, palliatieve zorg. Meestal zijn de klachten subtiel, maar soms heftiger.",
    },
    {
        "category": "Mobiliteit",
        "examples": """- Vandaag geholpen met de passieve lift. Dit ging goed.
- Veel rondgelopen vandaag. Mw vergeet steeds haar rollator.
- De banden van de rolstoel zijn zacht. Kan de fysio hier naar kijken?
- De transfers gaan steeds moeilijker. Mw hangt erg in de actieve lift. Glijdt weg. Wil graag nog met de actieve lift geholpen worden, maar dit gaat eigenlijk niet meer. @ Ergo, graag je advies.
""",
        "note_topics": "loophulpmiddelen, de rolstoel, valgevaar, valincidenten, transfers, tilliften. De meeste rapportages gaan over dagelijkse dingetjes, dus niet alles is een ernstig incident.",
    },
]

# response = llm.generate([[prompt]])

# print(response.generations[0][0].text)

# Resultatenlijst om rapporten op te slaan
results = []

# Loop over de input data en genereer rapporten
for input_data in input_data_list:
    # Maak de prompt met de input_data
    prompt = prompt_template.format(**input_data)

    # Genereer de respons van het LLM
    response = llm.generate([[prompt]])
    results.append(response)

print((results[1]).generations[0][0].text)
