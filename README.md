# slm-itgc-audit
Small Language Model for ITGC Auditability (Phase 1)”.
# =====================================================
# SLM ITGC — Pipeline complet (Option A: seuil LIME dynamique)
# (sans HF datasets / pyarrow) — 
# =====================================================

# ---------- Imports & setup ----------
import os, json, hashlib, platform, sys, random, glob, warnings, time
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from IPython.display import display

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, accuracy_score, f1_score

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding,
    TextClassificationPipeline
)

# ---------- Config & seeds ----------
SEED = 42
MAX_LEN = 256
N_EPOCHS = 8
TRAIN_BS = 16
EVAL_BS  = 32

# LIME (tu peux laisser 200, la génération est limitée au test set)
N_LIME = 200
LIME_SAMPLES = 2000

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

MODEL_NAME = "prajjwal1/bert-tiny"
LABELS = ["Conforme", "Non conforme", "Partiel"]
label2id = {l:i for i,l in enumerate(LABELS)}
id2label = {i:l for l,i in label2id.items()}

BASE_DIR  = "./slm_itgc_runs"
FINAL_DIR = "./slm_itgc_final"
os.makedirs(BASE_DIR, exist_ok=True)
os.makedirs(FINAL_DIR, exist_ok=True)

# ⚠️ ADAPTE ICI ton chemin CSV
CSV_PATH = "/content/itgc_gestion_acces.csv"
assert os.path.exists(CSV_PATH), f"CSV introuvable: {CSV_PATH}"

print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | Python: {platform.python_version()}")
print("Labels:", label2id)

# ---------- SHA du dataset & chargement CSV ----------
with open(CSV_PATH, "rb") as f:
    CSV_SHA256 = hashlib.sha256(f.read()).hexdigest()

df_raw = pd.read_csv(CSV_PATH, sep=";")
rename_map = {}
if "Texte" in df_raw.columns: rename_map["Texte"] = "text"
if "Label enrichi" in df_raw.columns: rename_map["Label enrichi"] = "label"
if "Norme / Référence" in df_raw.columns: rename_map["Norme / Référence"] = "reference"
df_raw = df_raw.rename(columns=rename_map)
assert {"text","label"}.issubset(df_raw.columns), f"Colonnes requises manquantes. Colonnes: {df_raw.columns.tolist()}"

# déduplication stricte
df = df_raw.drop_duplicates(subset=["text"]).reset_index(drop=True)

def map_to_3cls(x):
    x_low = str(x).lower().strip()
    if x_low.startswith("conforme"):       return "Conforme"
    if x_low.startswith("non conforme"):   return "Non conforme"
    if "partiel" in x_low:                 return "Partiel"
    return None

df["mapped"] = df["label"].apply(map_to_3cls)
bad = df[df["mapped"].isna()]
if not bad.empty:
    bad_path = os.path.join(FINAL_DIR, "labels_non_reconnus.csv")
    bad[["text","label"]].to_csv(bad_path, index=False)
    raise ValueError(f"{len(bad)} étiquette(s) non reconnue(s). Corrigez {bad_path} puis relancez.")
df["label"] = df["mapped"]; df.drop(columns=["mapped"], inplace=True)
df["label_id"] = df["label"].map(label2id)
df = df[["text","label","label_id"]]

print("Aperçu:"); display(df.head(3))
print("\nRépartition classes:"); print(df["label"].value_counts())

# ---------- Split ----------
train_df, test_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label_id"]
)
train_model = train_df[["text","label_id"]].rename(columns={"label_id":"labels"}).reset_index(drop=True)
test_model  = test_df[["text","label_id"]].rename(columns={"label_id":"labels"}).reset_index(drop=True)

# ---------- Tokenizer & encodage ----------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_df(df_):
    enc = tokenizer(
        df_["text"].tolist(),
        truncation=True, max_length=MAX_LEN,
        padding=False, return_tensors=None
    )
    labels = df_["labels"].to_numpy()
    return enc, labels

enc_train, y_train = tokenize_df(train_model)
enc_test,  y_test  = tokenize_df(test_model)

# Dataset PyTorch léger
class TxtClsDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings; self.labels = labels
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

train_ds = TxtClsDataset(enc_train, y_train)
test_ds  = TxtClsDataset(enc_test,  y_test)
collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ---------- Modèle, entraînement ----------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=len(LABELS), id2label=id2label, label2id=label2id
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        "f1_micro": f1_score(labels, preds, average="micro"),
    }

training_args = TrainingArguments(
    output_dir=os.path.join(BASE_DIR, "hf_outputs"),
    num_train_epochs=N_EPOCHS,
    per_device_train_batch_size=TRAIN_BS,
    per_device_eval_batch_size=EVAL_BS,
    learning_rate=2e-4,
    weight_decay=0.01,
    seed=SEED,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    tokenizer=tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics
)

trainer.train()

# ---------- Évaluation & exports ----------
eval_out = trainer.evaluate()
print("Résultats globaux :", eval_out)

preds = trainer.predict(test_ds)
logits = preds.predictions
y_pred = logits.argmax(-1)
y_true = y_test

# Rapport CSV + JSON
report_dict = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True, digits=2, zero_division=0)
pd.DataFrame(report_dict).T.to_csv(os.path.join(FINAL_DIR, "classification_report.csv"), encoding="utf-8")
with open(os.path.join(FINAL_DIR, "classification_report.json"), "w", encoding="utf-8") as f:
    json.dump(report_dict, f, ensure_ascii=False, indent=2)

# Matrice de confusion PNG + CSV
cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABELS))))
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABELS).plot(values_format="d")
plt.title("Matrice de confusion (test)"); plt.tight_layout()
plt.savefig(os.path.join(FINAL_DIR, "confusion_matrix.png"), dpi=150); plt.close()
pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(os.path.join(FINAL_DIR, "confusion_matrix.csv"), encoding="utf-8")

# ---------- Prédictions + proba + SHA ----------
probas = torch.softmax(torch.tensor(logits), dim=1).numpy()
pred_df = pd.DataFrame({
    "id": np.arange(len(y_true)),
    "label_true": [LABELS[i] for i in y_true],
    "label_pred": [LABELS[i] for i in y_pred],
})
for i, cls in enumerate(LABELS):
    pred_df[f"proba_{cls}"] = probas[:, i]
pred_csv = os.path.join(FINAL_DIR, "predictions.csv"); pred_df.to_csv(pred_csv, index=False, encoding="utf-8")
with open(pred_csv, "rb") as f:
    Path(os.path.join(FINAL_DIR, "predictions.sha256.txt")).write_text(
        hashlib.sha256(f.read()).hexdigest()+"\n", encoding="utf-8"
    )

# ---------- LIME (explications = min(N_LIME, len(test_df))) ----------
from lime.lime_text import LimeTextExplainer
from transformers import TextClassificationPipeline
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x  # fallback si tqdm absent

pipe = TextClassificationPipeline(
    model=trainer.model,
    tokenizer=tokenizer,
    return_all_scores=True,
    function_to_apply="softmax",
    truncation=True, max_length=MAX_LEN,
    batch_size=32
)
def predict_proba(texts):
    outputs = pipe(texts)
    ordered = []
    for out in outputs:
        scores = {d["label"]: d["score"] for d in out}
        ordered.append([scores.get(f"LABEL_{i}", 0.0) for i in range(len(LABELS))])
    return np.array(ordered)

explainer = LimeTextExplainer(class_names=LABELS, random_state=SEED)
lime_dir = os.path.join(FINAL_DIR, "lime_html"); os.makedirs(lime_dir, exist_ok=True)
rng = np.random.RandomState(SEED)
n_target = min(N_LIME, len(test_df))
sample_idx = rng.choice(len(test_df), size=n_target, replace=False)

for idx in tqdm(sample_idx, desc=f"Génération LIME (n={n_target})"):
    text = test_df.iloc[idx]["text"]
    out_path = os.path.join(lime_dir, f"lime_{idx}.html")
    if os.path.exists(out_path):  # reprise possible
        continue
    exp = explainer.explain_instance(
        text_instance=text,
        classifier_fn=predict_proba,
        num_features=10,
        num_samples=LIME_SAMPLES
    )
    exp.save_to_file(out_path)
    time.sleep(0.01)

# ---------- Reproductibilité & versioning ----------
pd.DataFrame(trainer.state.log_history).to_csv(os.path.join(FINAL_DIR, "training_log.csv"), index=False, encoding="utf-8")
args_dict = trainer.args.to_dict()
with open(os.path.join(FINAL_DIR, "training_args.json"), "w", encoding="utf-8") as f:
    json.dump(args_dict, f, ensure_ascii=False, indent=2)
Path(os.path.join(FINAL_DIR, "seed.txt")).write_text(str(SEED)+"\n", encoding="utf-8")

manifest = {
    "timestamp_utc": datetime.utcnow().isoformat()+"Z",
    "python": sys.version, "platform": platform.platform(),
    "cuda_available": torch.cuda.is_available(), "torch": torch.__version__,
    "transformers": __import__("transformers").__version__,
    "pandas": pd.__version__, "numpy": np.__version__,
    "scikit_learn": __import__("sklearn").__version__,
    "model_name": MODEL_NAME, "labels": LABELS,
}
Path(os.path.join(FINAL_DIR, "manifest.json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

# Hash des scripts .py (entêtes même si vide)
hash_rows = []
for p in glob.glob("**/*.py", recursive=True):
    try:
        with open(p, "rb") as fh:
            h = hashlib.sha256(fh.read()).hexdigest()
        hash_rows.append({"path": p, "sha256": h})
    except Exception:
        pass
script_hash_csv = os.path.join(FINAL_DIR, "script_hashes.csv")
if hash_rows:
    pd.DataFrame(hash_rows).to_csv(script_hash_csv, index=False, encoding="utf-8")
else:
    pd.DataFrame(columns=["path","sha256"]).to_csv(script_hash_csv, index=False, encoding="utf-8")

# Données gelées, SHA, distribution, mapping
dataset_copy = os.path.join(FINAL_DIR, "dataset.csv")
Path(dataset_copy).write_bytes(Path(CSV_PATH).read_bytes())
Path(os.path.join(FINAL_DIR, "dataset.sha256.txt")).write_text(CSV_SHA256+"\n", encoding="utf-8")

class_dist = df["label"].value_counts().rename_axis("Classe").reset_index(name="Nombre")
class_dist["Pourcentage"] = (class_dist["Nombre"]/len(df)*100).round(2)
cat = pd.CategoricalDtype(categories=["Conforme","Non conforme","Partiel"], ordered=True)
class_dist["Classe"] = class_dist["Classe"].astype(cat)
class_dist = class_dist.sort_values("Classe").reset_index(drop=True)
class_dist.to_csv(os.path.join(FINAL_DIR, "class_distribution.csv"), index=False, encoding="utf-8")

if {"text","reference"}.issubset(df_raw.columns):
    df_raw[["text","reference"]].drop_duplicates().rename(columns={"text":"assertion"}).to_csv(
        os.path.join(FINAL_DIR, "assertion_reference_map.csv"), index=False, encoding="utf-8"
    )

# ---------- Proof Harness (3 exigences) — Option A dynamique ----------
def read_text(p): return Path(p).read_text(encoding="utf-8").strip()
def sha256_file(p):
    with open(p, "rb") as f: return hashlib.sha256(f.read()).hexdigest()
def approx_equal(a,b,tol=1e-6): return abs(float(a)-float(b))<=tol
def check_exists(paths,miss):
    ok=True
    for p in paths:
        if not Path(p).exists(): miss.append(p); ok=False
    return ok

def prove_traceability():
    notes, missing, ok = [], [], True
    pred_csv = os.path.join(FINAL_DIR, "predictions.csv")
    pred_sha = os.path.join(FINAL_DIR, "predictions.sha256.txt")
    lime_dir = os.path.join(FINAL_DIR, "lime_html")
    ok &= check_exists([pred_csv, pred_sha, lime_dir], missing)
    if ok:
        ok_sha = (sha256_file(pred_csv) == read_text(pred_sha).split()[0]); ok &= ok_sha
        notes.append(f"[Traçabilité] SHA256(predictions.csv) = {ok_sha}")
        dfp = pd.read_csv(pred_csv)
        expected = {"id","label_true","label_pred"} | {f"proba_{c}" for c in LABELS}
        ok_cols = expected.issubset(set(dfp.columns)); ok &= ok_cols
        notes.append(f"[Traçabilité] Colonnes prédictions = {ok_cols}")
        # ✅ Exigence dynamique : au moins une explication par prédiction (bornée à 200)
        required = min(200, len(dfp))
        n_lime = len(list(glob.glob(os.path.join(lime_dir, '*.html'))))
        ok_lime = n_lime >= required
        notes.append(f"[Traçabilité] LIME HTML (>={required}) = {ok_lime} (found {n_lime})")
        ok &= ok_lime
    else:
        notes.append(f"[Traçabilité] Manquants: {missing}")
    return ok, notes

def prove_auditability():
    notes, missing, ok = [], [], True
    pred_csv = os.path.join(FINAL_DIR, "predictions.csv")
    rep_csv = os.path.join(FINAL_DIR, "classification_report.csv")
    rep_json = os.path.join(FINAL_DIR, "classification_report.json")
    cm_csv  = os.path.join(FINAL_DIR, "confusion_matrix.csv")
    ok &= check_exists([pred_csv, rep_csv, rep_json, cm_csv], missing)
    if ok:
        dfp = pd.read_csv(pred_csv)
        y_true = dfp["label_true"].astype(str).values
        y_pred = dfp["label_pred"].astype(str).values
        rep_recalc = classification_report(y_true, y_pred, target_names=LABELS, output_dict=True, zero_division=0)
        rep_csv_df = pd.read_csv(rep_csv, index_col=0)
        keys = [("macro avg","precision"), ("macro avg","recall"), ("macro avg","f1-score")]
        ok_met = True
        for idx, met in keys:
            v_csv  = float(rep_csv_df.loc[idx, met]) if met in rep_csv_df.columns else None
            v_calc = float(rep_recalc[idx][met])
            if v_csv is None or not approx_equal(v_csv, v_calc, tol=1e-4):
                ok_met = False; notes.append(f"[Auditabilité] Mismatch {idx}/{met}: saved={v_csv} vs recalculated={v_calc}")
        ok &= ok_met; notes.append(f"[Auditabilité] Classification report cohérent = {ok_met}")
        cm_saved = pd.read_csv(cm_csv, index_col=0)
        cm_calc = pd.DataFrame(confusion_matrix(y_true, y_pred, labels=LABELS), index=LABELS, columns=LABELS)
        ok_cm = cm_saved.equals(cm_calc); ok &= ok_cm
        notes.append(f"[Auditabilité] Matrice de confusion identique = {ok_cm}")
    else:
        notes.append(f"[Auditabilité] Manquants: {missing}")
    return ok, notes

def prove_reproducibility():
    notes, missing, ok = [], [], True
    seed_file   = os.path.join(FINAL_DIR, "seed.txt")
    args_json   = os.path.join(FINAL_DIR, "training_args.json")
    manifest_js = os.path.join(FINAL_DIR, "manifest.json")
    data_copy   = os.path.join(FINAL_DIR, "dataset.csv")
    data_sha    = os.path.join(FINAL_DIR, "dataset.sha256.txt")
    scr_hashes  = os.path.join(FINAL_DIR, "script_hashes.csv")
    class_dist  = os.path.join(FINAL_DIR, "class_distribution.csv")
    ok &= check_exists([seed_file, args_json, manifest_js, data_copy, data_sha, scr_hashes, class_dist], missing)
    if ok:
        ok_seed = (int(read_text(seed_file).split()[0]) == SEED); ok &= ok_seed
        notes.append(f"[Reproductibilité] Seed == {SEED} = {ok_seed}")
        with open(args_json, "r", encoding="utf-8") as f: args_obj = json.load(f)
        ok_args_seed = (args_obj.get("seed", None) == SEED); ok &= ok_args_seed
        notes.append(f"[Reproductibilité] TrainingArguments.seed == {SEED} = {ok_args_seed}")
        with open(manifest_js, "r", encoding="utf-8") as f: mani = json.load(f)
        must = ["python","platform","torch","transformers","pandas","numpy","scikit_learn","model_name","labels"]
        ok_manifest = all(k in mani and str(mani[k])!="" for k in must); ok &= ok_manifest
        notes.append(f"[Reproductibilité] Manifest champs clés présents = {ok_manifest}")
        ok_data_sha = (sha256_file(data_copy) == read_text(data_sha).split()[0]); ok &= ok_data_sha
        notes.append(f"[Reproductibilité] SHA256(dataset.csv) = {ok_data_sha}")
        cdf = pd.read_csv(class_dist)
        ok_classes = set(["Conforme","Non conforme","Partiel"]).issubset(set(cdf["Classe"].astype(str))); ok &= ok_classes
        notes.append(f"[Reproductibilité] class_distribution couvre 3 classes = {ok_classes}")
        # au moins 1 script hashé
        try:
            sh = pd.read_csv(scr_hashes)
            ok_sh = {"path","sha256"}.issubset(sh.columns) and (len(sh) >= 1)
        except pd.errors.EmptyDataError:
            ok_sh = False
        ok &= ok_sh
        notes.append(f"[Reproductibilité] script_hashes.csv valide (>=1) = {ok_sh}")
    else:
        notes.append(f"[Reproductibilité] Manquants: {missing}")
    return ok, notes

def run_proof_suite():
    os.makedirs(FINAL_DIR, exist_ok=True)
    results, notes = {}, []
    t_ok, t_notes = prove_traceability(); notes += t_notes
    a_ok, a_notes = prove_auditability(); notes += a_notes
    r_ok, r_notes = prove_reproducibility(); notes += r_notes
    results["Traçabilité"]=t_ok; results["Auditabilité"]=a_ok; results["Reproductibilité"]=r_ok
    md = ["# SLM ITGC — Proof Report\n","## Résumé\n"]
    for k,v in results.items(): md.append(f"- **{k}** : {'✅ OK' if v else '❌ NON VERIFIÉ'}")
    md.append("\n## Détails\n"); md += [f"- {line}" for line in notes]
    Path(os.path.join(FINAL_DIR, "proof_report.md")).write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    # assert all(results.values()), "Au moins une exigence n'est pas satisfaite — voir proof_report.md"

run_proof_suite()
print("\n✅ Terminé. Livrables dans:", FINAL_DIR)


PyTorch: 2.8.0+cu126 | CUDA: False | Python: 3.12.11
Labels: {'Conforme': 0, 'Non conforme': 1, 'Partiel': 2}
Aperçu:
text	label	label_id
0	Les droits d’accès sont revus tous les 6 mois.	Conforme	0
1	Aucune revue des droits depuis 18 mois.	Non conforme	1
2	La revue des droits est effectuée de manière i...	Partiel	2



Répartition classes:
label
Conforme        324
Non conforme    218
Partiel         218
Name: count, dtype: int64
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at prajjwal1/bert-tiny and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
 [304/304 00:23, Epoch 8/8]
Step	Training Loss
Résultats globaux : {'eval_loss': 0.30711159110069275, 'eval_accuracy': 0.9144736842105263, 'eval_f1_macro': 0.9150326797385621, 'eval_f1_micro': 0.9144736842105263, 'eval_runtime': 0.1564, 'eval_samples_per_second': 971.703, 'eval_steps_per_second': 31.964, 'epoch': 8.0}
Device set to use cpu
Génération LIME (n=152): 100%|██████████| 152/152 [00:00<00:00, 13123.39it/s]# SLM ITGC — Proof Report

## Résumé

- **Traçabilité** : ✅ OK
- **Auditabilité** : ✅ OK
- **Reproductibilité** : ✅ OK

## Détails

- [Traçabilité] SHA256(predictions.csv) = True
- [Traçabilité] Colonnes prédictions = True
- [Traçabilité] LIME HTML (>=152) = True (found 152)
- [Auditabilité] Classification report cohérent = True
- [Auditabilité] Matrice de confusion identique = True
- [Reproductibilité] Seed == 42 = True
- [Reproductibilité] TrainingArguments.seed == 42 = True
- [Reproductibilité] Manifest champs clés présents = True
- [Reproductibilité] SHA256(dataset.csv) = True
- [Reproductibilité] class_distribution couvre 3 classes = True
- [Reproductibilité] script_hashes.csv valide (>=1) = True

✅ Terminé. Livrables dans: ./slm_itgc_final

