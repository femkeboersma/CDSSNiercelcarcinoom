"""Everything to do with the model: loading it and making a prediction.

You can test this file on its own, without the web app:
    python prediction.py
"""
from pathlib import Path

import numpy as np

MODEL_FILE = Path('/Users/femkeboersma/Downloads/cdss_streamlit_demo-main/model.keras').parent / "model.keras"

# The model expects its 8 inputs in exactly this order (the column order of
# the training data). These are also the column names in the database.
FEATURES = [
    "leeft",
    "gesl",
    "tumsoort",
    "diag_basis",
    "gedrag",
    "tum_afm",
    "chemo",
    "rt",
    "org_chir",
    "pos_lymf"
]


def load_model():
    """Load your trained model from model.keras."""
    import keras  # imported here so the rest of this file works without TensorFlow

    return keras.models.load_model(str(MODEL_FILE), compile=False)


def preprocess(x):
    """Transform the raw input values the same way as during training.

    x is an array with one row of raw values (in FEATURES order) per patient.
    By default nothing happens. If your model was trained on transformed inputs
    (scaled, encoded, ...) and that step is NOT inside the model itself, do it here.
    Everything in this app goes through this function, so the app and your
    checks always see the same inputs.
    """
    return x


def make_input(patient, vit_stat=None):
    """Turn a patient (a dict with the FEATURES as keys) into the model's input.

    If vit_stat is given, it replaces the patient's stored value
    for this prediction only (nothing is written to the database).
    """
    values = dict(patient)
    if vit_stat is not None:
        values["vit_stat"] = vit_stat
    name = values.get("display_name", "this patient")
    for feature in FEATURES:
        if feature not in values:
            raise ValueError(f"No value called '{feature}' for {name}. "
                             "Do FEATURES and the database columns have the same names?")
        if values[feature] is None:
            raise ValueError(f"{name} has no value for '{feature}' (it is empty in the database). "
                             "The model needs a number for every input.")
    raw = np.array([[float(values[feature]) for feature in FEATURES]])
    return preprocess(raw)


def check_model(model, X):
    """Check that the model fits this app. Returns a problem description, or None if OK.

    X holds example inputs made with make_input, one row per patient.
    """
    try:
        out = np.asarray(model.predict(X, verbose=0))
    except Exception as error:
        n = len(FEATURES)
        return (f"The model could not make a prediction from {n} input values. "
                f"Does it expect exactly {n} inputs, one for each name in FEATURES? (Error: {error})")
    if out.shape != (len(X), 1):
        return (f"The model should return one number per patient, i.e. an output of shape "
                f"({len(X)}, 1) for {len(X)} patients, but it returned shape {out.shape}.")
    if not np.all((out >= 0) & (out <= 1)):
        return ("The model returned values outside the range 0 to 1, "
                "so its output cannot be read as a probability.")
    return None


def get_prediction(model, patient, vit_stat=None):
    """Return the model's output for one patient: a number between 0 and 1."""
    x = make_input(patient, vit_stat)
    return float(model.predict(x, verbose=0)[0, 0])


if __name__ == "__main__":
    # Checkpoint: the five patients from datafase.sql.
    examples = [dict(zip(FEATURES, values)) for values in [
        [20, 1, 711100, 7, 2, 12, 2, 'C508', 55],   # Timmy Tim Tim Leipzig
        [51, 2, 711310, 2, 3, 3, 1, 'C500', 0],
        [48, 2, 452358, 5, 3, 10, 2, 'C501', 19],
        [30, 1, 452355, 6, 3, 25, 0, 'C502', 20],
        [90, 2, 711100, 7, 2, 32, 4, 'C508', 35],

    ]]
    model = load_model()
    problem = check_model(model, np.vstack([make_input(p) for p in examples]))
    if problem:
        raise SystemExit("Problem with model.keras: " + problem)
    timmy = examples[0]
    print("Model input: ", make_input(timmy))
    print("Prediction:  ", round(get_prediction(model, timmy),3))
    print("Met blijven leven = Ja:", round(get_prediction(model, timmy, vit_stat=0), 3))
