def lees_bestand(bestandsnaam):
    with open(bestandsnaam, 'r') as file:
        inhoud = file.read()
    return inhoud

def schrijf_bestand(bestandsnaam, tekst):
    with open(bestandsnaam, 'w') as file:
        file.write(tekst)

#Deze functie maakt verbinding met de OpenAI API en genereert een 'response' gebaseerd op de rollen en inhoud die worden meegegeven (s_role en u_role). De functie maakt gebruik van het meegegeven model, en biedt de mogelijkheid om de seed (voor reproduceerbaarheid) en het aantal antwoorden (n) te bepalen.
def genereer_zorgdata(s_role, u_role, model='gpt-3.5-turbo', seed=None, n=1):
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": s_role},
            {"role": "user", "content": u_role}
        ],
        seed=seed,
        n=n
    )
    return completion