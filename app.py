"""A toy clinical decision support app.

Run:  streamlit run app.py

How Streamlit works, in one sentence: every time the user clicks or types
something, Streamlit runs this whole script again from top to bottom and
redraws the page. Things that must survive between runs (like a value the
user typed) are kept in st.session_state.

This is a technical demo. It is NOT meant to be used for clinical decisions.
"""
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import streamlit as st

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
        0: "alleen overlijdensverklaring beschikbaar",
        1: "alleen klinisch onderzoek (anamnese en lichamelijk onderzoek)",
        2: "klinisch-diagnostische onderzoeken, exploratieve chirurgie of obductie (zonder microscopisch bevestiging)",
        4: "specifieke biochemische en/of immunologische laboratoriumonderzoeken",
        5: "hematologische (beenmergcytologie = bijvoorbeeld beenmergpunctie, bloeduitstrijk) of cytologische bevestiging van de primaire tumor of metastasen, of microscopische bevestiging zeker, maar onduidelijk of dit cytologie of histologie betreft",
        6: "histologische bevestiging uitsluitend van metastase(n), inclusief bij obductie",
        7: "histologische bevestiging van de primaire tumor, of onduidelijk of histologische bevestiging de primaire tumor of een metastase betreft, obductie (met histologische bevestiging)"
    },
}

#Titel van de pagina
st.set_page_config(page_title="Prognose hulp voor patiënten met een niercelcarcinoom", page_icon=":hospital:", layout="centered")
st.title("Prognose hulp voor het gesprek met mantelzorgers van patiënten met een niercelcarcinoom")
st.caption("Technische demo. Niet bedoeld voor klinische beslissingen.")

# controleert of de benodigde bestanden bestaan, anders stopt de app met een foutmelding
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


def load_patients():
    """Read all patients from the database, as {id: {column: value}}."""
    with closing(sqlite3.connect(DB_FILE)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute("SELECT * FROM patient ORDER BY id").fetchall()
    return {row["id"]: dict(row) for row in rows}


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



# Je kunt hier een patient kiezen uit een dropdown, en de gegevens van die patient worden getoond.
patient_id = st.selectbox(
    "Kies een patiënt",
    options=list(patients),
    format_func=lambda pid: patients[pid]["display_name"],
)
current_patient = patients[patient_id]

resultaat = st.container()  # wordt hieronder gevuld, maar staat bovenaan de pagina

# Gegevens van de patiënt in gewoon Nederlands
st.subheader("Gegevens van de patiënt")
weergave = {}
for kolom, label in LABELS.items():
    waarde = current_patient[kolom]
    waarde = CODES.get(kolom, {}).get(waarde, waarde)
    weergave[label] = str(waarde)
st.table({"Gegeven": list(weergave.keys()), "Waarde": list(weergave.values())})

# Velden die de huisarts mag aanpassen: sleutel -> (label, minimum, maximum)
VELDEN = {
    "leeft": ("Leeftijd (jaar)", 0, 120),
    "tum_afm": ("Tumorgrootte (mm)", 0, 999),
    "pos_lymf": ("Aantal positieve lymfeklieren", 0, 99),
}


def toepassen(pid):
    """Submit-knop: bewaar de ingevulde waarden voor deze patiënt."""
    st.session_state[f"toegepast_{pid}"] = {
        k: st.session_state[f"{k}_{pid}"] for k in VELDEN
    }


def resetten(pid, patient):
    """Reset-knop: zet de velden terug op de waarden uit de database."""
    for k in VELDEN:
        st.session_state[f"{k}_{pid}"] = int(patient[k])
    st.session_state.pop(f"toegepast_{pid}", None)


# Startwaarden uit de database (alleen als het veld nog niet bestaat)
for k in VELDEN:
    st.session_state.setdefault(f"{k}_{patient_id}", int(current_patient[k]))

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
    p = float(prediction.get_prediction(model, aangepast, None))
    return p if KANS_IS_OVERLIJDEN else 1 - p

# Kans met de oorspronkelijke en met de aangepaste waarden, om te vergelijken
kans_orig = float(prediction.get_prediction(model, current_patient, None))
kans_orig = kans_orig if KANS_IS_OVERLIJDEN else 1 - kans_orig
kans_nu = voorspel_kans_overlijden(aangepast)

#kennisgebaseerde deel
with resultaat:
    st.subheader(f"Gesprek met de mantelzorger van {current_patient['display_name']}")

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

        st.subheader("Advies")
        for a in r.adviezen:
            if a.code == "E":
                st.success(f"**{a.conclusie}**")
                st.markdown(pictogram_html(r.n_in_leven), unsafe_allow_html=True)
                if aangepast != current_patient:
                    st.warning(
                        f"Let op: dit is berekend met aangepaste waarden. "
                        f"Kans op overlijden: {kans_nu:.1%} (met de oorspronkelijke "
                        f"waarden: {kans_orig:.1%})."
                    )
                # if aangepast != current_patient:
                #     st.warning("Let op: dit is berekend met aangepaste waarden.")
                st.info(a.advies)
            else:
                st.info(f"**{a.conclusie}**\n\n{a.advies}")
            st.caption(f"Advies {a.code} - Waarom: {a.reden} - Bron: {a.bron}")

            with st.expander("Hoe komt dit advies tot stand?"):
                st.write("Route door de flowchart: " + " → ".join(r.pad))
