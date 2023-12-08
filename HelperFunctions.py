def lees_bestand(bestandsnaam):
    with open(bestandsnaam, 'r') as file:
        inhoud = file.read()
    return inhoud

def schrijf_bestand(bestandsnaam, tekst):
    with open(bestandsnaam, 'w') as file:
        file.write(tekst)