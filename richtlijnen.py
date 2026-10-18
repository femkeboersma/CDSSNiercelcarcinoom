""" Kennisgebaseerde onderdeel - Van huisarts naar mantelzorger

Deze module volgt de aangeleverde flowchart van het CDSS. Elke beslisknop heeft een eigen code (k1-k4)
en elk advies heeft dit ook (A-F).

    K1 toestemming patiënt om info met mantelzorger te delen?
        Nee --> A
    K2 kan de mantelzorger nu informatie ontvangen? (beslissing huisarts)
        Nee --> B
    K3 wil de mantelzorger informatie over de prognose?
        Nee --> C --> F
    K4 zijn alle modelvariabelen bekend?
        Nee --> D --> F
        ja --> neuraal netwerk --> E --> F

Bronnen:
    Richtlijnen Niercelcarcinoom (NVU), https://richtlijnendatabase.nl/richtlijn/niercelcarcinoom/startpagina_niercelcarcinoom.html
    - 'Voorlichting en communicatie na stellen diagnose'
        - 'diagnose en prognose'
        - 'besluitvorming'

    K1/A (toestemming) volgt uit het beroepsgeheim (WGBO), niet uit deze richtlijn.
    K4/D is een eigen ontwerpkeuze (geen voorspelling zonder volledige invoer).

"""

from dataclasses import dataclass, field
import math

# Invoervariabelen van het neurale netwerk (zelfde volgorde/namen als in het model)
LABELS = {
    "leeft": "Leeftijd (jaar)",
    "gesl": "Geslacht",
    "tumsoort": "Tumorsoort (code)",
    "diag_basis": "Basis van de diagnose",
    "gedrag": "Gedrag van de tumor",
    "tum_afm": "Tumorgrootte (mm)",
    "later": "Zijde",
    "topo_sublok": "Lokalisatie (code)",
    "pos_lymf": "Aantal positieve lymfeklieren",
}

# Waarden die als 'onbekend' gelden.

ONBEKEND = {None, "", "onbekend", "Onbekend"}

BRON_DIAG = "Richtlijn Niercelcarcinoom - Voorlichting en communicatie na stellen diagnose"
BRON_PROGNOSE = "Richtlijn Niercelcarcinoom - Diagnose en prognose"
BRON_BESLUIT = "Richtlijn Niercelcarcinoom - Besluitvorming"
BRON_WGBO = "WGBO / beroepsgeheim arts (KNMG)"
EIGEN_KEUZE = "Eigen ontwerpkeuze CDSS"

@dataclass
class Advies:
    code: str        # A-F, zelfde als in de flowchart
    conclusie: str   # wat het systeem concludeert
    advies: str      # wat de huisarts moet doen
    reden: str       # welke gegevens tot deze conclusie leidden
    bron: str

@dataclass
class Resultaat:
    pad: list[str] = field(default_factory=list)
    adviezen: list[Advies] = field(default_factory=list)
    kans_overlijden: float | None = None               # uitkomst neuraal netwerk (0-1)
    n_in_leven: int | None = None                      # van de 100, na 5 jaar
    ontbrekend: list[str] = field(default_factory=list)

def is_onbekend(waarde) -> bool:
    if isinstance(waarde, float) and math.isnan(waarde):
        return True
    return waarde in ONBEKEND


def ontbrekende_variabelen(patient: dict) -> list[str]:
    return [naam for naam in LABELS if is_onbekend(patient.get(naam))]

def advies_F() -> Advies:
    return Advies(
        code="F",
        conclusie="Afronding van het gesprek.",
        advies=("Informeer de mantelzorger over de vervolgstappen en over beschikbare informatie (bijv. de KWF-brochure Nierkanker en de Vereniging Leven met blaas- of nierkanker). Leg het gesprek vast in het HIS."),
        reden="Gesprek met mantelzorger heeft plaatsgevonden.",
        bron=BRON_DIAG,
    )

def kennisgebaseerd_advies(patient: dict,
                           toestemming: bool,
                           kan_ontvangen: bool,
                           wil_prognose: bool,
                           voorspel_kans) -> Resultaat:
    """
    patient        : dict met de labels (zie LABELS)
    toestemming    : K1 - patiënt heeft toestemming gegeven om te delen
    kan_ontvangen  : K2 - mantelzorger kan nu informatie ontvangen (besluit huisarts)
    wil_prognose   : K3 - mantelzorger wil informatie over de prognose
    voorspel_kans  : functie(patient) -> kans op overlijden binnen 5 jaar (0-1);
                     dit is het neurale netwerk en wordt alleen aangeroepen bij K4 = ja
    """
    r = Resultaat()

    # K1 - Toestemming
    if not toestemming:
        r.pad.append("K1:nee")
        r.adviezen.append(Advies(
            code="A",
            conclusie="Er is geen toestemming van de patiënt geregistreerd.",
            advies=("Deel geen medische informatie. Bespreek met de patiënt of hij/zij "
                    "informatie met de mantelzorger wil delen. Bied de mantelzorger wel "
                    "steun en een luisterend oor."),
            reden="Toestemming patiënt: nee.",
            bron=BRON_WGBO,
        ))
        return r
    r.pad.append("K1:ja")

    # K2 - Kan de mantelzorger nu informatie ontvangen?
    if not kan_ontvangen:
        r.pad.append("K2:nee")
        r.adviezen.append(Advies(
            code="B",
            conclusie="De mantelzorger kan op dit moment geen informatie ontvangen.",
            advies=("Geef eerst ruimte aan de emotionele reactie. Geef nu geen informatie "
                    "en plan een vervolgconsult."),
            reden="Huisarts beoordeelt: mantelzorger kan nu geen informatie ontvangen.",
            bron=f"{BRON_PROGNOSE}; {BRON_DIAG}",
        ))
        return r
    r.pad.append("K2:ja")

    # K3 - Wil de mantelzorger prognose-informatie?
    if not wil_prognose:
        r.pad.append("K3:nee")
        r.adviezen.append(Advies(
            code="C",
            conclusie="De mantelzorger wil nu geen informatie over de prognose.",
            advies=("Leg dit vast in het dossier en bied aan er later op terug te komen."),
            reden="Mantelzorger wil prognose-informatie: nee.",
            bron=BRON_BESLUIT,
        ))
        r.adviezen.append(advies_F())
        return r
    r.pad.append("K3:ja")

    # K4 - Zijn alle modelvariabelen bekend?
    r.ontbrekend = ontbrekende_variabelen(patient)
    if r.ontbrekend:
        r.pad.append("K4:nee")
        lijst = ", ".join(LABELS[v] for v in r.ontbrekend)
        r.adviezen.append(Advies(
            code="D",
            conclusie="Geen individuele voorspelling mogelijk.",
            advies=f"Vul de ontbrekende gegevens in: {lijst}. Vraag deze op bij de specialist.",
            reden=f"Ontbrekende modelvariabelen: {lijst}.",
            bron=EIGEN_KEUZE,
        ))
        r.adviezen.append(advies_F())
        return r
    r.pad.append("K4:ja")

    # Neuraal netwerk
    r.kans_overlijden = float(voorspel_kans(patient))
    r.n_in_leven = 100 - round(r.kans_overlijden * 100)
    r.pad.append("NN")

    r.adviezen.append(Advies(
        code="E",
        conclusie=(f"Van de 100 mensen met vergelijkbare kenmerken zijn er na 5 jaar naar "
                   f"verwachting {r.n_in_leven} in leven."),
        advies=("Bespreek dit met de mantelzorger in deze vorm (aantallen, niet "
                "percentages) en gebruik het pictogram. Controleer daarna of het begrepen "
                "is (teach-back). Benoem het voorbehoud: dit is gebaseerd op vergelijkbare "
                "patiënten en niet alle omstandigheden zijn meegenomen."),
        reden=f"Alle modelvariabelen bekend. Modelschatting: {r.n_in_leven} van de 100 in leven.",
        bron=f"{BRON_BESLUIT}; {BRON_PROGNOSE}; {BRON_DIAG}",
    ))
    r.adviezen.append(advies_F())
    return r

def pictogram_html(n_in_leven: int, kleur_leven: str = "#2e8b57",
                   kleur_overleden: str = "#c8c8c8") -> str:
    """100 poppetjes (10 x 10): n_in_leven gekleurd, de rest grijs."""

    def poppetje(kleur: str) -> str:
        return (f'<svg width="22" height="30" viewBox="0 0 22 30" style="margin:1px">'
                f'<circle cx="11" cy="6" r="5" fill="{kleur}"/>'
                f'<path d="M3 30 V17 a8 8 0 0 1 16 0 V30 Z" fill="{kleur}"/></svg>')

    poppetjes = [poppetje(kleur_leven if i < n_in_leven else kleur_overleden)
                 for i in range(100)]
    rijen = ["".join(poppetjes[r * 10:(r + 1) * 10]) for r in range(10)]
    return ('<div style="font-family:sans-serif;line-height:0">' + "<br>".join(rijen) +
            f'<p style="margin-top:8px;line-height:1.4">{n_in_leven} van de 100 in leven na 5 jaar '
            f'(gekleurd), {100 - n_in_leven} overleden (grijs)</p></div>')
