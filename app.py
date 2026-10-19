"""Een CDSS die berekent wat de kans is dat een patiënt met een niercelcarcinoom overlijdt, en geeft advies over het gesprek met de mantelzorger voor de huisarts.
Run:  streamlit run app.py
Dit is een technische demo. Het is niet bedoeld voor gebruik in de klinische praktijk.
"""
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import streamlit as st
import pandas as pd

import prediction
from richtlijnen import kennisgebaseerd_advies, pictogram_html

DB_FILE = Path(__file__).parent / "ehr.db"
# Kans op overlijden of overleving, afhankelijk van wat het model voorspelt. Als het model de kans op overlijden voorspelt, dan is KANS_IS_OVERLIJDEN = True. Als het model de kans op overleving voorspelt, dan is KANS_IS_OVERLIJDEN = False.
KANS_IS_OVERLIJDEN = True

# Geeft beschrijving bij de kolomnamen in de database, en bij de codes in sommige kolommen.
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
CODES = {
    "gesl": {1: "Man", 2: "Vrouw"},
    "gedrag": {2: "In situ", 3: "Kwaadaardig"},
    "later": {1: "Links", 2: "Rechts"},
    "diag_basis": {
        0: "Er is alleen een overlijdensverklaring beschikbaar",
        1: "Er is tot op heden alleen klinisch onderzoek (anamnese en lichamelijk onderzoek) uitgevoerd",
        2: "Er is al klinisch-diagnostisch onderzoek uitgevoerd, bijvoorbeeld: exploratieve chirurgie of obductie (maar zonder microscopisch bevestiging)",
        4: "Er zijn specifieke biochemische en/of immunologische laboratoriumonderzoeken uitgevoerd",
        5: "Er is hematologische of cytologische bevestiging van de primaire tumor of metastasen, of er is microscopische bevestiging, maar nog onduidelijk of dit cytologie of histologie betreft",
        6: "Er is histologische bevestiging uitsluitend van metastase(n), inclusief bij obductie",
        7: "Er is histologische bevestiging van de primaire tumor, of onduidelijk of histologische bevestiging de primaire tumor of een metastase betreft, obductie (met histologische bevestiging)"
    },
}
# Velden die de huisarts mag aanpassen, voor elke waarde geldt tussen de haakjes: (label, minimum, maximum)
VELDEN = {
    "leeft": ("Leeftijd (jaar)", 0, 120),
    "tum_afm": ("Tumorgrootte (mm)", 0, 999),
    "pos_lymf": ("Aantal positieve lymfeklieren", 0, 99),
}
AFRONDING_CODES = {"F"}


# Titel van de pagina, :green[] zorgt ervoor dat de tekst in het oranje wordt weergegeven.
st.set_page_config(page_title="Prognosegesprek met de mantelzorger", page_icon=":hospital:", layout="wide")
st.title(":green[Prognosegesprek] met de mantelzorger van een patiënt met een niercelcarcinoom")
st.caption("Technische demo. Niet bedoeld voor klinische beslissingen.")
st.divider()  # scheidingslijn onder de kop

# Controleert of de benodigde bestanden bestaan, anders stopt de app met een foutmelding
if not DB_FILE.exists():
    st.error("ehr.db not found. Run `python setup_database.py` first (see the README).")
    st.stop()
if not prediction.MODEL_FILE.exists():
    st.error("model.keras not found. Train your model and save it as model.keras "
             "in the same folder as app.py (see the README).")
    st.stop()


# Laadt het model eenmalig en de patienten opnieuw bij elke run
@st.cache_resource(max_entries=1)  # laadt het model een keer, ipv bij elke muisklik
def get_model(file_modified_time):  # laadt het model een keer opnieuw wanneer model.keras wordt aangepast
    return prediction.load_model()

# Laadt de patienten uit de database, als {id: {column: value}}. De patienten worden opnieuw geladen bij elke run, zodat de app altijd de laatste versie van de database gebruikt.
def load_patients():
    with closing(sqlite3.connect(DB_FILE)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM patient ORDER BY id").fetchall()
    return {row["id"]: dict(row) for row in rows}

# error management: als er geen patienten zijn, of als het model niet kan worden geladen, of als de patienten niet passen bij het model, dan wordt er een foutmelding getoond en stopt de app.
patients = load_patients()
if not patients:
    st.warning("The database has no patients. Check database.sql and run setup_database.py.")
    st.stop()

try:
    model = get_model(prediction.MODEL_FILE.stat().st_mtime)
except ImportError as error:
    st.error("TensorFlow is not installed in the Python environment that runs this app. "
             "Activate your virtual environment and run `pip install -r requirements.txt`. "
             f"(Error: {error})")
    st.stop()
except Exception as error:
    st.error("model.keras exists but could not be loaded. Was it saved with "
             "`model.save('model.keras')`, using the same TensorFlow version as in "
             f"requirements.txt? See Troubleshooting in the README. (Error: {error})")
    st.stop()

try:
    all_inputs = np.vstack([prediction.make_input(p) for p in patients.values()])
except ValueError as error:
    st.error(f"Problem with the patient data: {error}")
    st.stop()

problem = prediction.check_model(model, all_inputs)
if problem:
    st.error("Your model does not fit this app yet. " + problem)
    st.stop()



# Welke patiënt is gekozen? None = nog niemand, dan toont het programma het welkomstscherm
st.session_state.setdefault("patient_id", None)

# Start-knop, het programma onthoudt de gekozen patiënt in st.session_state.patient_id
def start_gesprek():
    st.session_state.patient_id = st.session_state.keuze_welkom

# Knop rechtsboven, leid de gebruiker terug naar het welkomstscherm.
def kies_andere_patient():
    st.session_state.patient_id = None
    st.session_state.pop("keuze_welkom", None)


# Welkomstscherm
if st.session_state.patient_id is None:
    st.subheader("Welkom")
    st.write("Met deze tool kunt u zich voorbereiden op het gesprek met de mantelzorger "
             "van een patiënt met een niercelcarcinoom. Kies hieronder een patiënt om te beginnen.")
    keuze = st.selectbox(
        "Kies een patiënt",
        options=list(patients),
        index=None,
        placeholder="Selecteer een patiënt...",
        format_func=lambda pid: patients[pid]["display_name"],
        key="keuze_welkom",
    )
    st.button("Start", type="primary", disabled=keuze is None, on_click=start_gesprek)
    st.stop()  # de rest van de pagina wordt pas getoond na het kiezen

# Hoofdscherm
patient_id = st.session_state.patient_id
current_patient = patients[patient_id]

_, knop_kolom = st.columns([3, 1])
knop_kolom.button("Andere patiënt kiezen", type="primary", on_click=kies_andere_patient, use_container_width=True)

# Twee kolommen: links de patiëntgegevens, rechts het gesprek en de "Wat als"-velden
links, midden, rechts = st.columns([1, 2, 1])

with midden:
    resultaat = st.container(border=True)  # wordt hieronder gevuld, maar staat bovenaan in het midden

with rechts:
    zijblok = st.container(border=True)  # geeft mogelijkheid voor opstart voor het gesprek en afronding van het gesprek als ondersteuning voor de huisarts.

# Gegevens van de patiënt in gewoon Nederlands
weergave = {}
for kolom, label in LABELS.items():
    waarde = current_patient[kolom]
    waarde = CODES.get(kolom, {}).get(waarde, waarde)
    weergave[label] = str(waarde)

with links:
    with st.container(border=True):
        st.subheader(":green[Gegevens van de patiënt]")
        tabel = pd.DataFrame({"Gegeven": weergave.keys(), "Waarde": weergave.values()})
        st.table(tabel.set_index("Gegeven"))

# Beschrijft de functies van de knoppen in het "Wat als"-formulier.
# De ingevulde waarden worden bewaard in st.session_state, zodat ze bij een volgende run van de app nog beschikbaar zijn.
# Submit-knop, zorgt ervoor dat de ingevulde waarden voor de patient bewaard blijven in st.session_state, hierdoor kunnen ze worden gebruikt in de "nieuwe voorspelling".
def toepassen(pid):
    st.session_state[f"toegepast_{pid}"] = {
        k: st.session_state[f"{k}_{pid}"] for k in VELDEN
    }

# reset-knop, herstelt de waarden van de patiënt naar de oorspronkelijke waarden uit de database, en verwijdert de aangepaste waarden uit st.session_state
def resetten(pid, patient):
    for k in VELDEN:
        st.session_state[f"{k}_{pid}"] = int(patient[k])
    st.session_state.pop(f"toegepast_{pid}", None)

# Startwaarden uit de database (alleen als het veld nog niet bestaat)
for k in VELDEN:
    st.session_state.setdefault(f"{k}_{patient_id}", int(current_patient[k]))

# with midden verzorgt de uitlijning van de "Wat als"-velden en de knoppen, zodat ze onder elkaar staan in het midden van de pagina.
with midden:
    with st.expander("Wat als de waarden anders zijn? (optioneel)", expanded=True):
        st.caption("Pas de waarden aan en klik op 'Bereken opnieuw'. "
                "Er wordt niets opgeslagen in de database.")
        with st.form(f"wat_als_{patient_id}"):
            for k, (label, minimum, maximum) in VELDEN.items():
                st.number_input(label, min_value=minimum, max_value=maximum,
                                key=f"{k}_{patient_id}")
            knop1, knop2 = st.columns(2)
            knop1.form_submit_button("Bereken opnieuw", type="primary",
                                    on_click=toepassen, args=(patient_id,))
            knop2.form_submit_button("Terug naar oorspronkelijke waarden",
                                    on_click=resetten, args=(patient_id, current_patient))

# Alleen de toegepaste waarden gaan naar het model
aangepast = dict(current_patient)
aangepast.update(st.session_state.get(f"toegepast_{patient_id}", {}))

# Voorspelling berekenen en bovenaan tonen
def voorspel_kans_overlijden(patient):
    p = float(prediction.get_prediction(model, patient, None))
    return p if KANS_IS_OVERLIJDEN else 1 - p

# Kans met de oorspronkelijke en met de aangepaste waarden, om te vergelijken
kans_orig = float(prediction.get_prediction(model, current_patient, None))
kans_orig = kans_orig if KANS_IS_OVERLIJDEN else 1 - kans_orig
kans_nu = voorspel_kans_overlijden(aangepast)

# kennisgebaseerde deel, in het midden
with resultaat:
    st.subheader(f":green[Gesprek met de mantelzorger van] {current_patient['display_name']}")

    def ja_nee(vraag, sleutel):
        antwoord = st.radio(vraag, ["Ja", "Nee"], index=None, horizontal=True, key=f"{sleutel}_{patient_id}")
        return None if antwoord is None else antwoord == "Ja"

    toestemming = ja_nee("Heeft de patiënt toestemming gegeven om medische informatie met mantelzorger te delen?", "k1")
    kan_ontvangen = wil_prognose = None
    if toestemming:
        kan_ontvangen = ja_nee("Kan de mantelzorger op dit moment informatie ontvangen? (Uw oordeel als huisarts)", "k2")
    if toestemming and kan_ontvangen:
        wil_prognose = ja_nee("Wil de mantelzorger informatie over de prognose?", "k3")

    # advies wordt pas getoond als hele flowchart is beantwoord
    klaar = (toestemming is False or (toestemming and kan_ontvangen is False) or (toestemming and kan_ontvangen and wil_prognose is not None))

    if not klaar:
        st.caption("Beantwoord de vragen hierboven om het advies te zien.")
    else:
        r = kennisgebaseerd_advies(patient=aangepast, toestemming=toestemming, kan_ontvangen=bool(kan_ontvangen), wil_prognose=bool(wil_prognose), voorspel_kans=voorspel_kans_overlijden)

        st.subheader(":green[Advies]")
        for a in r.adviezen:
            if a.code in AFRONDING_CODES:
                continue  # de afronding staat in het rechterblok
            if a.code == "E":
                st.success(f"**{a.conclusie}**")
                st.markdown(pictogram_html(r.n_in_leven), unsafe_allow_html=True)
                if aangepast != current_patient:
                    st.warning(
                        f"Let op: dit is berekend met aangepaste waarden. "
                        f"Kans op overlijden: {kans_nu:.1%} (met de oorspronkelijke "
                        f"waarden: {kans_orig:.1%})."
                    )
                st.info(a.advies)
            else:
                st.info(f"**{a.conclusie}**\n\n{a.advies}")
            st.caption(f"Advies {a.code} - Waarom: {a.reden} - Bron: {a.bron}")

# Rechterblok: eerst de opstart, na het advies de afronding
# Rechterblok: eerst de opstart, daarna de afronding die bij de antwoorden past
with zijblok:
    if not klaar:
        st.subheader(":green[Opstart van het gesprek]")
        st.write("- Zorg voor een rustige plek en voldoende tijd.\n"
                 "- Controleer of de patiënt toestemming heeft gegeven voor het delen van medische informatie met de mantelzorger.\n"
                 "- Vraag eerst wat de mantelzorger al weet. \n"
                 "- Doorloop de vragen op het formulier")
    else:
        st.subheader(":green[Afronding van het gesprek]")

        # De volgorde van de checks volgt de volgorde van de vragen
        if toestemming is False:
            st.write("- Deel geen medische informatie met de mantelzorger.\n"
                     "- Luister wel naar de mantelzorger en erken diens zorgen.\n"
                     "- Bespreek met de patiënt of toestemming later mogelijk is.")
        elif kan_ontvangen is False:
            st.write("- Geef nu geen informatie, maar laat weten dat er later ruimte voor is.\n"
                     "- Plan een vervolggesprek.\n"
                     "- Vraag wat de mantelzorger nu nodig heeft en wijs op ondersteuning.")
        elif wil_prognose is False:
            st.write("- Respecteer dat de mantelzorger geen informatie over de prognose wil.\n"
                     "- Laat weten dat de mogelijkheid blijft bestaan om hier later op terug te komen.\n"
                     "- Bespreek wat de mantelzorger zelf nodig heeft en wijs op ondersteuning.")
        else:
            st.write("- Vraag wat de mantelzorger van de informatie heeft begrepen.\n"
                     "- Geef ruimte voor emoties en vragen.\n"
                     "- Maak afspraken voor een vervolggesprek en wijs op ondersteuning.")

        with st.expander("Hoe komt dit advies tot stand?"):
            st.write("Route door de flowchart: " + " → ".join(r.pad))