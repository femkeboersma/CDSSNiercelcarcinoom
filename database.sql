-- A tiny "electronic health record" with five (fictional) patients.
-- The values are the 8 inputs the model expects, in the same order as the
-- training data: pregnancies, glucose, diastolic blood pressure, skin
-- thickness, insulin, BMI, diabetes pedigree function, age.
--
-- This file is written for SQLite. Run `python setup_database.py` to turn it
-- into the database file ehr.db.

DROP TABLE IF EXISTS patient;

CREATE TABLE patient (
  id                  INTEGER PRIMARY KEY,
  display_name        TEXT NOT NULL,
  leeft               INTEGER,
  gesl                INTEGER,
  tumsoort            INTEGER,
  diag_basis          INTEGER,
  gedrag              INTEGER,
  tum_afm             INTEGER,
  later               INTEGER,
  topo_sublok         CHAR(4),
  pos_lymf            INTEGER
);

INSERT INTO patient
  (display_name, leeft, gesl, tumsoort, diag_basis, gedrag, tum_afm, later, topo_sublok, pos_lymf)
VALUES
  ('Timmy Tim Tim Leipzig',  20, 1, 711100, 7, 2, 12, 2, 'C508', 55),
  ('Roderick van Aszelt',    51, 2, 711310, 2, 3, 3, 1, 'C500', 0),
  ('Jackie Bal',             48, 2, 452358, 5, 3, 10, 2, 'C501', 19),
  ('Kees de Krentebol',      30, 1, 452355, 6, 3, 25, 2, 'C502', 20),
  ('Lie Ting',               90, 2, 711100, 7, 2, 32, 1, 'C508', 35);
