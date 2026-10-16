# A toy clinical decision support app, built with Streamlit

This repo contains a small demo for students: a web app that lets a user

1. pick a patient from a tiny (fictional) health record,
2. optionally type in a different blood pressure value, and
3. see what a machine-learning model predicts about that patient's chance of diabetes.

The app is ready to go, except for one thing: **the model. Training it is your job.**
After that, you will use this repo as the starting point for your own app to diagnose cancer
(see [Part 2](#part-2-build-your-own-app-to-diagnose-cancer)).

This is a [Streamlit](https://streamlit.io) version of
[Python_ML_web_app_demo](https://github.com/ace-dvm/Python_ML_web_app_demo) (Flask + PythonAnywhere),
with the same data and the same five patients.

> ⚠️ **This is a technical demo. It must not be used to make clinical decisions.**

---

## What's in this repo

| File | What it does |
|---|---|
| `app.py` | The web app (the page you see in the browser). |
| `prediction.py` | Loads your model, checks it, and turns a patient into a prediction. |
| `database.sql` | The patient records, written in SQL. |
| `setup_database.py` | Turns `database.sql` into the database file `ehr.db`. |
| `ehr.db` | The database file (already built for you). |
| `requirements.txt` | The Python packages the app needs. |
| `model.keras` | **Not included: you train and add this yourself** (see below). |

## How the pieces fit together

```
 database.sql ──(setup_database.py)──▶ ehr.db ─────────┐
                                                       ▼
                                  app.py ◀──▶ prediction.py ◀── model.keras
                                    ▲                              ▲
                                    │                              │
                    you, in the browser                  your own training code
```

When you click something in the app, Streamlit re-runs `app.py` from top to bottom:
it reads the patients from `ehr.db`, builds the 8 numbers the model needs, asks the
model for a prediction and redraws the page.

### Why is there a database?

Strictly speaking, an app that only takes values from the user and returns a prediction doesn't
need a database. So why is there one?

The database plays the role of the hospital's **electronic health record (EHR)**. In a real clinical
decision support system, the clinician doesn't type in the patient's data: lab results, vital signs
and history are already in the record. The system reads them from there, and the clinician only
selects a patient and, at most, corrects or adds a value. That is what this app does: it reads the
patient from `ehr.db`, and the user can change the blood pressure for the prediction
(nothing is written back to the database).

This matters for three reasons:

- **Decision support works best when it fits into the clinician's workflow.** A tool that makes
  clinicians retype data that already exists elsewhere tends not to get used.
- **Typing is error-prone.**
- **The data and the model are designed separately.** The database has its own column names,
  units and missing values. Getting from "database column" to "model input, in the right order and
  with the right preprocessing" is a real integration problem, and you will meet it in Part 2.

`ehr.db` is a very small stand-in. A real EHR is a separate system that applications reach through
a standard interface.

---

## Your task: train the model

Train a neural network that predicts diabetes, using the **Pima Indians Diabetes** dataset:
[pima-indians-diabetes.data.csv](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)
(768 patients, no header row; the 9th column is the outcome: 1 = diabetes, 0 = no diabetes).

You decide how to build and train it. To work in this app, your model must:

1. **Take 8 inputs, in this order:**
   pregnancies, glucose, diastolic blood pressure, skin thickness, insulin, BMI,
   diabetes pedigree function, age. This is the column order of the CSV, and the order
   of `FEATURES` in `prediction.py`.
2. **Accept the raw values** as they are in the database (e.g. glucose = 148, age = 50).
   If your model needs scaled inputs, either put the scaling *inside* the model (use a Keras
   `Normalization` layer, not a `Lambda` layer, which won't load), or do it in the
   `preprocess` function in `prediction.py`.
3. **Output one number between 0 and 1** for each patient: the model's estimate that the patient has diabetes.
4. **Be a Keras model saved as `model.keras`**, with `model.save("model.keras")`,
   and placed in the same folder as `app.py`. Don't rename a file saved in another format
   (such as `model.h5`): it won't load.

The app checks points 1 and 3 when it starts and tells you if something doesn't fit.
It cannot check point 2 or the order in point 1, so double-check those yourself:
**if the order or scaling is wrong, the app still shows a number, but it is the wrong number.**

You can train wherever you like in your machine (a script, a notebook) but make sure you do not upload the IKNL data on the cloud. Training in the same environment where you run the app (step 2 below) avoids version problems.

---

## Checklist: before you run the app

Most problems come from using the wrong Python. Go through this list once you have downloaded
the repo and have your `model.keras`. The detailed commands are in the next section.

- [ ] **Python 3.10–3.13 is installed.** Check with `python --version` (or `python3` / `py`).
  If your version is newer or older, install one in this range *next to* it; you don't need to remove the other.
- [ ] **You made one environment for this project, with that Python.** Either a virtual environment
  or a conda environment (see step 2 below). Using conda? If `(base)` is shown in your terminal,
  plain `python` is conda's Python, not the one you just installed.
- [ ] **Inside the activated environment, the version is right.** After activating it, `python --version`
  must show 3.10–3.13. If not, delete the environment and create it again with the right Python.
- [ ] **The packages are installed in that environment.** Run `pip install -r requirements.txt` with the
  environment active. Check with `python -c "import tensorflow, streamlit; print('OK')"`.
- [ ] **Your editor uses that environment.** If you work in an editor or IDE (VS Code, PyCharm,
  Spyder, Jupyter…), select the environment's Python as its *interpreter* (or *kernel*). Editors
  don't find environments you made in a terminal by themselves. To see the path to give it, run
  `python -c "import sys; print(sys.executable)"` with the environment active. After changing it,
  open a new terminal in the editor.
- [ ] **`model.keras` is in the same folder as `app.py`.**
- [ ] **You run the commands from the project folder**, with the environment active:
  first `python prediction.py` (it should print a prediction between 0 and 1), then `streamlit run app.py`.

---

## Run it on your own computer

You need **Python 3.10–3.13** (3.12 is a safe choice). TensorFlow does not support newer
versions yet, so check with `python --version` first.

> **`python`, `python3` or `py`?** The commands below use `python`. On macOS and Linux
> you may need `python3` instead; on Windows, `py`. Once the virtual environment is
> active (step 2), plain `python` works everywhere.

If something doesn't work, look at [Troubleshooting](#troubleshooting) at the end of this README.

**1. Get the code.** Click the green **Code** button on GitHub → **Download ZIP**, and unzip it.
Open a terminal in that folder.

**2. Install the packages** in a virtual environment (a private set of packages for this project).

On macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows (Command Prompt or PowerShell):

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

If you use Anaconda or Miniconda, you can make a conda environment instead (any system):

```bash
conda create -n dss-demo python=3.12
conda activate dss-demo
pip install -r requirements.txt
```

This downloads TensorFlow, which is big, so it can take a few minutes.
You need to activate the environment again (the second line) every time you open a new terminal.
Check that `python --version` now shows 3.10–3.13, and if you use an editor,
select this environment as its interpreter (see the [checklist](#checklist-before-you-run-the-app)).

**3. (Optional) Rebuild the database.** `ehr.db` is already there. If you change `database.sql`, run:

```bash
python setup_database.py
```

**4. Add your model.** Train it (see [Your task](#your-task-train-the-model)) and put `model.keras` in this folder.

**5. Check that the model works on its own** (before involving the web app):

```bash
python prediction.py
```

You should see the model's input for Susan Foreman and a prediction between 0 and 1.

**6. Start the app:**

```bash
streamlit run app.py
```

Your browser opens the app at http://localhost:8501. Stop it with `Ctrl+C` in the terminal.

Two things that look like problems but aren't:
- The very first time, Streamlit asks for your email address in the terminal. Just press Enter.
- TensorFlow prints many messages when it starts (about CUDA, GPUs or oneDNN).
  On a normal laptop these are harmless.

---

## Put it online (free)

You can host the app on [Streamlit Community Cloud](https://share.streamlit.io).

1. Put the files in your own GitHub repository, **including your `model.keras`**.
   (On GitHub: **Add file → Upload files**, then drag in the files *inside* the folder,
   not the folder itself, so that `app.py` sits at the top level of the repository.)
   Also upload the hidden file `.gitignore`: it stops your `.venv` folder from being added
   if you later use git. To see hidden files, press `Cmd+Shift+.` in the macOS Finder, or
   choose **View → Show → Hidden items** in Windows File Explorer.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app** → **Yup, I have an app**.
4. Choose your repository, branch `main`, and main file `app.py`.
5. Click **Advanced settings**, choose **the same Python version as your environment**, and click **Save**.
6. Click **Deploy**.

The first start takes a few minutes, because TensorFlow has to be installed. You then get a web address like https://<your-app-name>.streamlit.app that works from any device. 
Good to know:

- Who can use it: an app from a public GitHub repository is public: anyone with the link can use it. An app from a private repository is private; under Settings → Sharing you can invite specific people by email (you can have one private app at a time). Never put real patient data in the app.
- Several users at once: each visitor gets their own session, so one person's choices don't affect another's.
- Updates: when you change files in your GitHub repository, the app updates automatically. After replacing model.keras, click Manage app → Reboot app to be sure the new model is used.
- Sleep: apps that nobody visits for 12 hours go to sleep; the next visitor wakes them up (this takes about a minute).

---

## Things to think about (Pima app)

- How did you measure your model's accuracy, and on which data? Why is accuracy on the training data too optimistic?
  How accurate would a "model" be that always says "no diabetes"? (About 65% of patients in the data don't have diabetes.)
- The five patients in `ehr.db` are copied from the Pima data. If your model was trained on them, what does that mean for the predictions you see?
- In this dataset a value of 0 often means "not measured": almost half the patients have insulin = 0.
  Susan Foreman's insulin is 0. Did you do anything about this when training? What does your model think that 0 means?
- The prediction is a number between 0 and 1, not a yes/no. Where would you put the cut-off, and who should decide?
  Does 0.70 really mean "a 70% chance"?
- The form accepts blood pressures up to 400, but the highest diastolic blood pressure in the data is 122.
  Try some extreme values. What happens, and should the app warn the user?
- The data come from women aged 21 and older of Pima heritage
  ([Smith et al., 1988](https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/)). Would you use this model for other patients?

---

## Part 2: build your own app to diagnose cancer

Once the Pima app works and you understand what every file does, you will turn it into
**your own decision support app that helps diagnose a type of cancer**, using the cancer dataset
for your assignment. Start from a copy of this repo (your own new GitHub repository) and change it step by step.

The app's structure stays the same: a database of (fictional) patients, a trained model, and a page
that shows the model's output for the selected patient. Almost everything *inside* that structure changes.

### What you will need to change

These are the places that have to agree with each other. Finding out exactly *what* to change
in each one is part of the assignment.

1. **Your model.** Explore your dataset first: which features are there, in which units, are values
   missing, and how many patients have cancer? Then train and evaluate your network on data it has
   not seen, and save it as `model.keras`. Decide which outcome is 1 (for example: 1 = malignant)
   and make sure the app says the same thing.
2. **`database.sql`**: a `patient` table whose columns are your features, and a few patient records.
   Use made-up patients or rows you held out from training, and never real data that could identify a person.
   Then run `python setup_database.py` again.
3. **`FEATURES` in `prediction.py`**: your feature names, the same as the database columns,
   **in the same order as during training**.
4. **Preprocessing**: whatever you did to the inputs during training (scaling, encoding, filling in
   missing values), the app must do in exactly the same way: inside the model, or in the
   `preprocess` function in `prediction.py`.
5. **`app.py`**: the title and texts on the page, and how the result is shown: a probability, a label, or both?
6. **Let the clinician edit any value.** In the Pima app the user can only change the blood pressure.
   In your app, the clinician must be able to change **any** of the patient's values before asking for a
   prediction. The values from the database are the starting point; changes are used for the prediction
   only and are not saved. This is a change to both `app.py` and `prediction.py` (look at `make_input`).
   Your page must also:
   - show clearly **which values were changed** compared to the record, so it is visible what the
     prediction was based on;
   - give every field a **sensible minimum and maximum**, based on your training data;
   - show the **record's values again** when the clinician switches to another patient.
7. **The checkpoint in `prediction.py`** (the part under `if __name__ == "__main__":`): example patients
   from your own data, so that `python prediction.py` still tests your model.
8. **`requirements.txt`**: any extra packages you use.
9. **`README.md`**: describe your own app, your data and your model.

**Tip:** change one thing at a time and run the app after each step. The error messages tell you
where it breaks. Test your app with a patient whose diagnosis you know.

### Things to think about (your cancer app)

- **Diagnosis or prediction?** The Pima study was about forecasting the *onset* of diabetes. Your app
  is about a diagnosis *now*. Who uses the app, at which moment, and what do they do with the result?
- **Which mistake is worse?** Telling a patient with cancer that they are fine, or the other way around?
  Choose your cut-off with this in mind, and report sensitivity and specificity, not only accuracy.
  If a missed cancer is 10 times worse than a false alarm, should the cut-off still be 0.5?
- **Where do the inputs come from?** Who measures them, and can a clinician realistically type them in?
  What should the app do with values outside the range seen in the training data?
- **Can the result be trusted?** On which data did you evaluate your model? Does your dataset look like
  the patients the app would be used on?
- **What should the page show?** A bare number, a label, a warning? What would a clinician need to see
  before acting on it?

---

## Credits and license

Based on [ace-dvm/Python_ML_web_app_demo](https://github.com/ace-dvm/Python_ML_web_app_demo).
The data are the Pima Indians Diabetes dataset
([Smith et al., 1988](https://pmc.ncbi.nlm.nih.gov/articles/PMC2245318/)).

Like the original, this code is licensed under the GNU General Public License v3.0 (see `LICENSE`).

---

## Troubleshooting

Find the stage where things go wrong, then the message you see.

### Installing

| What you see | What to do |
|---|---|
| `python: command not found` or `'python' is not recognized` | Use `python3` (macOS / Linux) or `py` (Windows). |
| `No matching distribution found for tensorflow` | Your Python version is too new (or too old) for TensorFlow. Check with `python --version`. Install Python 3.12 and create the environment with it: `python3.12 -m venv .venv` (macOS / Linux) or `py -3.12 -m venv .venv` (Windows). |
| Windows: `running scripts is disabled on this system` when activating | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in the same PowerShell window and activate again, or use Command Prompt instead. |
| Windows: installing TensorFlow fails with an error about a file path or "No such file or directory" | Windows has a limit on path length. Move the project to a short path (e.g. `C:\ml\`) and start again, or [enable long paths](https://learn.microsoft.com/en-us/windows/win32/fileio/maximum-file-path-limitation). |
| `ModuleNotFoundError: No module named 'streamlit'` (or `keras`), or `'streamlit' is not recognized` | The virtual environment is not active. Activate it (step 2) and, if needed, run `pip install -r requirements.txt`. |
| `python --version` shows the wrong version *inside* the environment | The environment was made with a different Python (e.g. conda's). Delete it and create it again with Python 3.10–3.13, e.g. `python3.12 -m venv .venv` or `conda create -n dss-demo python=3.12`. |
| Your editor says "no interpreter", or uses a different Python than your terminal | Select the environment's Python as the editor's interpreter; see the [checklist](#checklist-before-you-run-the-app). Then open a new terminal in the editor. |

### Running the app

| What you see | What to do |
|---|---|
| Warnings about `missing ScriptRunContext`, and no app opens | You ran `python app.py`. Start the app with `streamlit run app.py`. |
| Streamlit asks for your email address | Press Enter to skip. This only happens the first time. |
| Many TensorFlow messages about CUDA, GPUs or oneDNN | Harmless on a normal laptop. Look for real errors in the app itself. |
| `ehr.db not found` or "The database has no patients" | Run `python setup_database.py` and check `database.sql`. |
| `Problem with the patient data: ... has no value for ...` | A value is empty (`NULL`) in `database.sql`. The model needs a number for every input. |
| `Problem with the patient data: No value called ...` | A name in `FEATURES` doesn't match a column in the database. |
| You replaced `model.keras` but the predictions didn't change | Refresh the page; the app reloads the model when the file changes. If that doesn't help, stop the app (`Ctrl+C`) and start it again. |

### Your model

| What you see | What to do |
|---|---|
| `model.keras not found` | Save your model as `model.keras` in the same folder as `app.py`. |
| `TensorFlow is not installed in the Python environment that runs this app` | Activate the virtual environment and run `pip install -r requirements.txt`. |
| `model.keras exists but could not be loaded` | Save it with `model.save("model.keras")`; don't rename a `.h5` file. Retrain with the TensorFlow version from `requirements.txt` (models from TensorFlow 2.15 or older may not load). If the error mentions a `Lambda` layer or `safe_mode`, do your scaling with a `Normalization` layer or in `preprocess` instead. |
| `Your model does not fit this app yet` | Read the message: it says which requirement (number of inputs, one output, values between 0 and 1) is not met. |
| The app runs, but the predictions look odd (the same for everyone, always close to 0 or 1) | Check the order of the inputs and the scaling: the app uses raw values in `FEATURES` order, plus whatever `preprocess` does. Compare with exactly what you did during training. |

### Online (Streamlit Community Cloud)

| What you see | What to do |
|---|---|
| Where are the error messages? | Open your app and click **Manage app** (bottom right) to see the logs. |
| The app can't find `app.py` or other files | You probably uploaded the folder instead of its contents. Either set the main file path to `foldername/app.py`, or upload the files again at the top level. |
| It works on your computer but not online | Your computer and the cloud have different TensorFlow versions. Run `python -c "import tensorflow as tf; print(tf.__version__)"` where you trained the model, and pin that version in `requirements.txt`, e.g. `tensorflow==2.21.0`. |
| You uploaded a new `model.keras` but the app still uses the old one | Click **Manage app → Reboot app**. |
| "This app has gone to sleep" | Click the button to wake it up; it takes a minute. |
