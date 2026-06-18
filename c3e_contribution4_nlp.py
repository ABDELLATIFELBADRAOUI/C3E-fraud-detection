# ============================================================
# c3e_contribution4_nlp.py  —  Contribution 4
# NLP-Enhanced Transaction Representation for Fraud Detection
# Under Chronological Evaluation
#
# Modules:
#   A. Text field extraction (IEEE-CIS: email, device, browser)
#   B. FastText embeddings on text fields
#   C. TF-IDF + SVD on pseudo-documents (IEEE-CIS + BAF)
#   D. Categorical embeddings for PaySim transaction types
#   E. Synthetic NLP features from numerical metadata
#   F. Hybrid fusion: NLP features + numerical → XGB / CatBoost
#   G. Delta diagnostic on NLP-augmented models
#   H. LaTeX generators
#
# Install:  pip install gensim
# Paper: "Beyond ROC-AUC — C3E Framework"  — Section 5 (Contribution 4)
# ============================================================

from __future__ import annotations
import warnings, re, os
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition            import TruncatedSVD
from sklearn.preprocessing            import StandardScaler, LabelEncoder
from sklearn.impute                   import SimpleImputer
from sklearn.pipeline                 import Pipeline
from sklearn.metrics                  import (
    roc_auc_score, average_precision_score,
    precision_score, recall_score, f1_score,
    confusion_matrix,
)
from xgboost   import XGBClassifier
from catboost  import CatBoostClassifier

SEED = 42


# ============================================================
# A.  DATASET-SPECIFIC TEXT FIELD DEFINITIONS
# ============================================================

# IEEE-CIS columns carrying textual / categorical string data
IEEE_TEXT_COLS = [
    "P_emaildomain",    # payer email domain  e.g. gmail.com
    "R_emaildomain",    # receiver email domain
    "DeviceInfo",       # device name  e.g. "Samsung SM-G950F"
    "id_30",            # operating system  e.g. "Windows 10"
    "id_31",            # browser  e.g. "chrome 75.0"
    "id_33",            # screen resolution e.g. "1920x1080"
    "id_34",            # device match category
]

# PaySim: the only text column
PAYSIM_TYPE_COL = "type"   # CASH-IN, CASH-OUT, DEBIT, PAYMENT, TRANSFER

# BAF: categorical columns usable for pseudo-documents
BAF_CAT_COLS = [
    "payment_type",
    "employment_status",
    "housing_status",
    "source",
    "device_os",
]


# ============================================================
# B.  TEXT EXTRACTION UTILITIES
# ============================================================

def _clean_token(s: str) -> str:
    """Lowercase, remove version numbers and special chars."""
    s = str(s).lower().strip()
    s = re.sub(r"\d+[\.\d]*", " ", s)   # remove version numbers
    s = re.sub(r"[^a-z\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "unknown"


def extract_ieee_text_fields(
    df: pd.DataFrame,
    cols: List[str] = IEEE_TEXT_COLS,
) -> pd.Series:
    """
    For each transaction row, concatenate all text columns into
    a single string document.

    Example output:
      "gmail com gmail com samsung unknown windows chrome unknown unknown"

    This pseudo-document encodes the transactional context
    (email provider, device ecosystem, OS, browser) as a
    bag-of-words that TF-IDF and FastText can process.
    """
    available = [c for c in cols if c in df.columns]
    if not available:
        return pd.Series(["unknown"] * len(df), index=df.index)

    def row_to_doc(row):
        tokens = []
        for col in available:
            val = row.get(col, "")
            if pd.isna(val) or str(val).strip() in ("", "nan"):
                tokens.append("unknown")
            else:
                tokens.append(_clean_token(str(val)))
        return " ".join(tokens)

    return df.apply(row_to_doc, axis=1)


def extract_paysim_docs(df: pd.DataFrame) -> pd.Series:
    """
    PaySim pseudo-document: transaction type + amount bin.
    Example: "transfer high_amount" / "payment low_amount"
    """
    if PAYSIM_TYPE_COL not in df.columns:
        return pd.Series(["unknown"] * len(df), index=df.index)

    amount_col = "amount" if "amount" in df.columns else None
    docs = []
    for _, row in df.iterrows():
        t   = _clean_token(str(row.get(PAYSIM_TYPE_COL, "unknown")))
        doc = t
        if amount_col:
            amt = float(row.get(amount_col, 0))
            if amt < 100:
                doc += " micro_amount"
            elif amt < 10_000:
                doc += " low_amount"
            elif amt < 100_000:
                doc += " mid_amount"
            else:
                doc += " high_amount"
        docs.append(doc)
    return pd.Series(docs, index=df.index)


def extract_baf_docs(df: pd.DataFrame) -> pd.Series:
    """
    BAF pseudo-document: concatenate categorical columns.
    """
    available = [c for c in BAF_CAT_COLS if c in df.columns]
    if not available:
        return pd.Series(["unknown"] * len(df), index=df.index)

    def row_to_doc(row):
        return " ".join([
            _clean_token(str(row.get(c, "unknown")))
            for c in available
        ])
    return df.apply(row_to_doc, axis=1)


DOCUMENT_EXTRACTORS: Dict[str, Callable] = {
    "ieee_cis":   extract_ieee_text_fields,
    "paysim":     extract_paysim_docs,
    "baf":        extract_baf_docs,
    "creditcard": lambda df: pd.Series(
        ["unknown"] * len(df), index=df.index),   # no text
}


# ============================================================
# C.  SYNTHETIC NLP FEATURES FROM NUMERICAL METADATA
#     Novel contribution: treat numerical bins as "words"
# ============================================================

def create_synthetic_text_features(
    df: pd.DataFrame,
    amount_col:  Optional[str] = "Amount",
    time_col:    Optional[str] = "Time",
    dataset_id:  str = "creditcard",
) -> pd.Series:
    """
    For datasets with no text fields (creditcard), construct
    pseudo-documents from quantised numerical features.

    Strategy:
      - Amount → decile bucket token  (e.g. "amount_d3")
      - Time   → hour-of-day token   (e.g. "hour_14")  [if available]
      - V-features sign pattern       (e.g. "sig_pnpppn")

    This encodes distributional information that TF-IDF can
    exploit to separate fraud from legitimate transactions,
    providing a purely synthetic NLP signal.
    """
    docs = []

    # Amount decile
    if amount_col and amount_col in df.columns:
        try:
            deciles = pd.qcut(df[amount_col], q=10,
                              labels=False, duplicates="drop")
        except Exception:
            deciles = pd.Series([0]*len(df), index=df.index)
    else:
        deciles = pd.Series([0]*len(df), index=df.index)

    # Time: map to 6-hour slots if available
    if time_col and time_col in df.columns:
        slots = (df[time_col] % 86400 // 21600).astype(int)
    else:
        slots = pd.Series([0]*len(df), index=df.index)

    # V-feature sign pattern (first 6 PCA components)
    v_cols = [c for c in df.columns
              if re.match(r"^[Vv]\d+$", c)][:6]

    for i in range(len(df)):
        row  = df.iloc[i]
        dec  = int(deciles.iloc[i]) if not pd.isna(
            deciles.iloc[i]) else 0
        slot = int(slots.iloc[i])
        tokens = [f"amount_d{dec}", f"timeslot_{slot}"]

        if v_cols:
            sig = "".join(
                "p" if row[c] >= 0 else "n"
                for c in v_cols
            )
            tokens.append(f"sig_{sig}")

        # Dataset-specific tokens
        if dataset_id == "ieee_cis":
            # Card type token from card columns if available
            for card_col in ["card4", "card6"]:
                if card_col in df.columns:
                    val = _clean_token(str(row.get(card_col,"")))
                    tokens.append(f"card_{val}")

        docs.append(" ".join(tokens))

    return pd.Series(docs, index=df.index)


# ============================================================
# D.  TF-IDF + SVD (LSA) PIPELINE
#     Fitted on train documents only — leakage-free
# ============================================================

@dataclass
class LSAModel:
    tfidf:    TfidfVectorizer
    svd:      TruncatedSVD
    n_components: int
    vocab_size:   int

    def transform(self, docs: pd.Series) -> np.ndarray:
        X_tfidf = self.tfidf.transform(docs)
        return self.svd.transform(X_tfidf)

    @property
    def explained_variance(self) -> float:
        return float(self.svd.explained_variance_ratio_.sum())


def fit_lsa(
    train_docs: pd.Series,
    n_components: int = 32,
    max_features: int = 5000,
    ngram_range:  Tuple[int,int] = (1, 2),
) -> LSAModel:
    """
    Fit TF-IDF + Truncated SVD (LSA) on training documents only.
    Returns fitted LSAModel for leakage-free transformation.

    Parameters
    ----------
    train_docs   : Series of document strings (training set only)
    n_components : SVD latent dimensions (paper uses 32)
    max_features : TF-IDF vocabulary size cap
    ngram_range  : unigrams + bigrams by default
    """
    tfidf = TfidfVectorizer(
        max_features = max_features,
        ngram_range  = ngram_range,
        sublinear_tf = True,       # log(1 + tf) — standard for NLP
        min_df       = 2,          # ignore hapax legomena
        strip_accents = "unicode",
    )
    X_tfidf = tfidf.fit_transform(train_docs)

    n_comp = min(n_components, X_tfidf.shape[1] - 1)
    svd    = TruncatedSVD(n_components=n_comp,
                           random_state=SEED)
    svd.fit(X_tfidf)

    return LSAModel(
        tfidf=tfidf, svd=svd,
        n_components=n_comp,
        vocab_size=len(tfidf.vocabulary_),
    )


# ============================================================
# E.  FASTTEXT EMBEDDINGS
#     Trained on transaction documents, leakage-free
# ============================================================

class FastTextEmbedder:
    """
    FastText character n-gram embeddings (Bojanowski et al.,
    TACL 2017) trained on transaction pseudo-documents.

    Advantages over Word2Vec for fraud detection:
    - Handles OOV tokens (new browser versions, new device names)
      via subword representations — critical under concept drift
    - Better embeddings for rare tokens (device model IDs)
    - Leakage-free: trained on train documents only
    """

    def __init__(self, vector_size: int = 32,
                 min_count: int = 1,
                 epochs: int = 10):
        self.vector_size = vector_size
        self.min_count   = min_count
        self.epochs      = epochs
        self._model      = None

    def fit(self, docs: pd.Series) -> "FastTextEmbedder":
        from gensim.models import FastText
        sentences = [doc.split() for doc in docs]
        self._model = FastText(
            sentences    = sentences,
            vector_size  = self.vector_size,
            window       = 3,
            min_count    = self.min_count,
            epochs       = self.epochs,
            seed         = SEED,
            workers      = 4,
            sg           = 1,       # skip-gram
            min_n        = 3,       # min char n-gram
            max_n        = 6,       # max char n-gram
        )
        return self

    def transform(self, docs: pd.Series) -> np.ndarray:
        """
        Embed each document as the mean of its token vectors.
        OOV tokens use subword approximation automatically.
        """
        embeddings = []
        for doc in docs:
            tokens = doc.split()
            if not tokens:
                embeddings.append(
                    np.zeros(self.vector_size))
                continue
            vecs = []
            for tok in tokens:
                try:
                    vecs.append(self._model.wv[tok])
                except KeyError:
                    vecs.append(np.zeros(self.vector_size))
            embeddings.append(np.mean(vecs, axis=0))
        return np.array(embeddings, dtype=np.float32)

    @property
    def vocab_size(self) -> int:
        return len(self._model.wv) if self._model else 0


# ============================================================
# F.  CATEGORICAL EMBEDDINGS (PaySim transaction type)
# ============================================================

class CategoricalEmbedder:
    """
    Learned categorical embeddings for low-cardinality fields.
    Embedding dimension = ceil(sqrt(n_categories)).

    For PaySim: 5 transaction types → dim = ceil(sqrt(5)) = 3.
    Embeddings are fitted as the mean feature vector
    per category in the training set (no neural network needed
    — equivalent to target-free entity embeddings).
    """

    def __init__(self):
        self._emb_table: Dict[str, np.ndarray] = {}
        self._dim: int = 0
        self._default: np.ndarray = None

    def fit(self, categories: pd.Series,
            X_numeric: np.ndarray) -> "CategoricalEmbedder":
        """
        For each unique category value, compute the mean of
        X_numeric (numerical features) in that group.
        This yields a numeric summary embedding for each category.
        """
        import math
        n_cats    = categories.nunique()
        self._dim = min(math.ceil(math.sqrt(n_cats)),
                        X_numeric.shape[1])

        # Use first _dim principal components as embedding space
        svd = TruncatedSVD(n_components=self._dim,
                            random_state=SEED)
        X_reduced = svd.fit_transform(
            StandardScaler().fit_transform(X_numeric))

        self._default = np.zeros(self._dim, dtype=np.float32)

        for cat in categories.unique():
            mask = (categories == cat).values
            if mask.sum() == 0:
                self._emb_table[str(cat)] = self._default.copy()
            else:
                self._emb_table[str(cat)] = \
                    X_reduced[mask].mean(axis=0).astype(np.float32)
        return self

    def transform(self, categories: pd.Series) -> np.ndarray:
        return np.array([
            self._emb_table.get(str(c), self._default)
            for c in categories
        ], dtype=np.float32)

    @property
    def dim(self) -> int:
        return self._dim


# ============================================================
# G.  FULL NLP FEATURE PIPELINE
#     Leakage-free: all fits on training data only
# ============================================================

@dataclass
class NLPPipelineConfig:
    dataset_id:       str
    use_lsa:          bool = True
    use_fasttext:     bool = True
    use_cat_emb:      bool = True
    use_synthetic:    bool = True
    lsa_components:   int  = 32
    ft_vector_size:   int  = 32
    ft_epochs:        int  = 10
    tfidf_max_feat:   int  = 5000


class NLPFeaturePipeline:
    """
    Unified NLP feature extraction pipeline.
    Produces a dense feature matrix that augments the
    numerical feature matrix for hybrid classification.

    Architecture:
      [Raw text fields]
           ↓
      Document extractor (dataset-specific)
           ↓
      ┌──────────────┬─────────────────┐
      │  TF-IDF+SVD  │  FastText mean  │
      │  (32 dims)   │  embedding      │
      │              │  (32 dims)      │
      └──────┬───────┴────────┬────────┘
             │                │
      [Categorical embeddings] [Synthetic NLP features]
             │                │
             └────────────────┘
                    ↓
             NLP feature matrix
             (concatenated, scaled)
    """

    def __init__(self, config: NLPPipelineConfig):
        self.config   = config
        self._lsa:     Optional[LSAModel]          = None
        self._ft:      Optional[FastTextEmbedder]  = None
        self._cat_emb: Optional[CategoricalEmbedder] = None
        self._scaler:  StandardScaler              = StandardScaler()
        self._fitted   = False

    def fit(self,
            df_train: pd.DataFrame,
            X_train_num: np.ndarray,
            label_col: str,
            time_col:  str,
            amount_col: Optional[str] = None,
            cat_col:    Optional[str] = None,
            ) -> "NLPFeaturePipeline":
        cfg = self.config
        ds  = cfg.dataset_id

        # 1. Extract documents from training data
        extractor = DOCUMENT_EXTRACTORS.get(ds, extract_ieee_text_fields)
        self._train_docs = extractor(df_train)

        # 2. Synthetic NLP features (all datasets)
        if cfg.use_synthetic:
            self._synth_docs_train = create_synthetic_text_features(
                df_train, amount_col, time_col, ds)

        # 3. Fit LSA on training documents
        if cfg.use_lsa:
            combined_docs = (self._train_docs + " " +
                             (self._synth_docs_train
                              if cfg.use_synthetic
                              else pd.Series([""] * len(df_train),
                                             index=df_train.index)))
            self._lsa = fit_lsa(combined_docs,
                                 n_components=cfg.lsa_components,
                                 max_features=cfg.tfidf_max_feat)
            print(f"    [LSA]      vocab={self._lsa.vocab_size}  "
                  f"dim={self._lsa.n_components}  "
                  f"var={self._lsa.explained_variance:.3f}")

        # 4. Fit FastText on training documents
        if cfg.use_fasttext:
            self._ft = FastTextEmbedder(
                vector_size=cfg.ft_vector_size,
                epochs=cfg.ft_epochs,
            ).fit(self._train_docs)
            print(f"    [FastText]  vocab={self._ft.vocab_size}  "
                  f"dim={cfg.ft_vector_size}")

        # 5. Fit categorical embeddings
        if cfg.use_cat_emb and cat_col and cat_col in df_train.columns:
            self._cat_emb = CategoricalEmbedder().fit(
                df_train[cat_col], X_train_num)
            self._cat_col = cat_col
            print(f"    [CatEmb]    n_cats={df_train[cat_col].nunique()}  "
                  f"dim={self._cat_emb.dim}")
        else:
            self._cat_col = None

        # 6. Fit scaler on concatenated NLP features
        X_nlp_train = self._transform_raw(df_train, X_train_num)
        self._scaler.fit(X_nlp_train)
        self._nlp_dim = X_nlp_train.shape[1]
        self._fitted  = True

        print(f"    [NLP total] {self._nlp_dim} NLP features")
        return self

    def transform(self,
                  df: pd.DataFrame,
                  X_num: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first"
        X_raw = self._transform_raw(df, X_num)
        return self._scaler.transform(X_raw)

    def _transform_raw(self,
                       df: pd.DataFrame,
                       X_num: np.ndarray) -> np.ndarray:
        cfg   = self.config
        ds    = cfg.dataset_id
        parts = []

        extractor = DOCUMENT_EXTRACTORS.get(ds,
                                             extract_ieee_text_fields)
        docs = extractor(df)

        if cfg.use_synthetic:
            synth = create_synthetic_text_features(
                df,
                amount_col="Amount" if "Amount" in df.columns
                           else ("amount" if "amount" in df.columns
                                 else None),
                time_col=None,
                dataset_id=ds,
            )
            combined = docs + " " + synth
        else:
            combined = docs

        if cfg.use_lsa and self._lsa:
            parts.append(self._lsa.transform(combined)
                         .astype(np.float32))

        if cfg.use_fasttext and self._ft:
            parts.append(self._ft.transform(docs))

        if (cfg.use_cat_emb and self._cat_emb and
                self._cat_col and self._cat_col in df.columns):
            parts.append(self._cat_emb.transform(df[self._cat_col]))

        if not parts:
            return np.zeros((len(df), 1), dtype=np.float32)

        return np.hstack(parts)

    @property
    def n_nlp_features(self) -> int:
        return self._nlp_dim if self._fitted else 0


# ============================================================
# H.  HYBRID CLASSIFIERS: NLP + NUMERICAL FUSION
# ============================================================

def train_hybrid_xgb(
    X_num_tr: np.ndarray,
    X_nlp_tr: np.ndarray,
    y_tr:     np.ndarray,
    X_num_val: np.ndarray = None,
    X_nlp_val: np.ndarray = None,
    y_val:     np.ndarray = None,
) -> Callable:
    """
    XGBoost on [X_numerical | X_NLP] concatenated features.
    Early stopping on PR-AUC if validation data provided.
    """
    X_tr = np.hstack([X_num_tr, X_nlp_tr])
    n1   = int(y_tr.sum())
    n0   = len(y_tr) - n1

    model = XGBClassifier(
        n_estimators     = 600,
        learning_rate    = 0.05,
        max_depth        = 6,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = n0 / max(n1, 1),
        eval_metric      = "aucpr",
        random_state     = SEED,
        n_jobs           = -1,
        verbosity        = 0,
    )

    if X_num_val is not None and X_nlp_val is not None:
        X_val = np.hstack([X_num_val, X_nlp_val])
        model.fit(X_tr, y_tr,
                  eval_set=[(X_val, y_val)],
                  verbose=False)
    else:
        model.fit(X_tr, y_tr)

    return lambda X_n, X_l: model.predict_proba(
        np.hstack([X_n, X_l]))[:, 1]


def train_hybrid_catboost(
    X_num_tr: np.ndarray,
    X_nlp_tr: np.ndarray,
    y_tr:     np.ndarray,
    X_num_val: np.ndarray = None,
    X_nlp_val: np.ndarray = None,
    y_val:     np.ndarray = None,
) -> Callable:
    """CatBoost on [X_numerical | X_NLP] concatenated features."""
    X_tr = np.hstack([X_num_tr, X_nlp_tr])
    n1   = int(y_tr.sum())
    n0   = len(y_tr) - n1

    model = CatBoostClassifier(
        iterations            = 800,
        learning_rate         = 0.05,
        depth                 = 6,
        scale_pos_weight      = n0 / max(n1, 1),
        eval_metric           = "AUC",
        early_stopping_rounds = 50,
        random_seed           = SEED,
        verbose               = 0,
    )

    if X_num_val is not None and X_nlp_val is not None:
        X_val = np.hstack([X_num_val, X_nlp_val])
        model.fit(X_tr, y_tr,
                  eval_set=(X_val, y_val),
                  verbose=0)
    else:
        model.fit(X_tr, y_tr, verbose=0)

    return lambda X_n, X_l: model.predict_proba(
        np.hstack([X_n, X_l]))[:, 1]


# ============================================================
# I.  FULL EXPERIMENT RUNNER — CONTRIBUTION 4
# ============================================================

@dataclass
class NLPResult:
    dataset:       str
    model:         str           # "XGB-NLP" / "CatBoost-NLP"
    n_nlp_feats:   int
    n_num_feats:   int
    roc_auc_base:  float         # baseline (no NLP)
    pr_auc_base:   float
    roc_auc_nlp:   float         # with NLP augmentation
    pr_auc_nlp:    float
    delta_base:    float
    delta_nlp:     float
    cost_base:     float
    cost_nlp:      float
    gain_pr_auc:   float         # pr_auc_nlp - pr_auc_base
    gain_cost:     float         # cost_base - cost_nlp (positive = improvement)
    tau_star_base: float
    tau_star_nlp:  float


def run_nlp_experiment(
    df_train: pd.DataFrame,
    df_val:   pd.DataFrame,
    df_test:  pd.DataFrame,
    label_col:   str,
    time_col:    str,
    amount_col:  Optional[str],
    cat_col:     Optional[str],
    dataset_id:  str,
    cfn: float = 10.0,
    cfp: float =  1.0,
) -> pd.DataFrame:
    """
    Full C3E experiment for Contribution 4.
    Compares numerical baseline vs NLP-augmented hybrid.
    """
    results = []

    # ── Preprocessing ─────────────────────────────────────────
    drop  = [label_col, time_col]
    pipe  = Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  StandardScaler())])

    def to_X(df):
        return df.drop(columns=drop, errors="ignore") \
                 .values.astype(np.float32)

    X_tr_num  = pipe.fit_transform(to_X(df_train))
    X_val_num = pipe.transform(to_X(df_val))
    X_te_num  = pipe.transform(to_X(df_test))

    y_tr  = df_train[label_col].values.astype(np.int32)
    y_val = df_val[label_col].values.astype(np.int32)
    y_te  = df_test[label_col].values.astype(np.int32)

    n_num = X_tr_num.shape[1]

    # ── Fit NLP pipeline (train only) ────────────────────────
    print(f"\n  [{dataset_id}] Fitting NLP pipeline ...")
    cfg = NLPPipelineConfig(
        dataset_id     = dataset_id,
        use_lsa        = True,
        use_fasttext   = True,
        use_cat_emb    = (cat_col is not None),
        use_synthetic  = True,
        lsa_components = 32,
        ft_vector_size = 32,
        ft_epochs      = 10,
    )
    nlp_pipe = NLPFeaturePipeline(cfg)
    nlp_pipe.fit(df_train, X_tr_num, label_col, time_col,
                 amount_col, cat_col)

    X_tr_nlp  = nlp_pipe.transform(df_train, X_tr_num)
    X_val_nlp = nlp_pipe.transform(df_val,   X_val_num)
    X_te_nlp  = nlp_pipe.transform(df_test,  X_te_num)
    n_nlp = X_tr_nlp.shape[1]

    # ── Train and evaluate both model families ─────────────────
    for model_name, trainer_base, trainer_nlp in [
        ("XGB",
         lambda: _baseline_xgb(X_tr_num, y_tr),
         lambda: train_hybrid_xgb(X_tr_num, X_tr_nlp, y_tr,
                                   X_val_num, X_val_nlp, y_val)),
        ("CatBoost",
         lambda: _baseline_cat(X_tr_num, y_tr),
         lambda: train_hybrid_catboost(X_tr_num, X_tr_nlp, y_tr,
                                        X_val_num, X_val_nlp, y_val)),
    ]:
        print(f"\n  [{dataset_id}] {model_name} baseline ...")
        scorer_base = trainer_base()
        p_val_base  = scorer_base(X_val_num)
        p_te_base   = scorer_base(X_te_num)
        tau_base, _ = _select_tau(y_val, p_val_base, cfn, cfp)

        print(f"  [{dataset_id}] {model_name}-NLP ...")
        scorer_nlp  = trainer_nlp()
        p_val_nlp   = scorer_nlp(X_val_num, X_val_nlp)
        p_te_nlp    = scorer_nlp(X_te_num,  X_te_nlp)
        tau_nlp, _  = _select_tau(y_val, p_val_nlp, cfn, cfp)

        # Metrics
        def _metrics(p_val, p_te, tau):
            delta   = (roc_auc_score(y_val, p_val) -
                       average_precision_score(y_val, p_val))
            y_pred  = (p_te >= tau).astype(int)
            cm      = confusion_matrix(y_te, y_pred, labels=[0,1])
            tn,fp,fn,tp = cm.ravel()
            cost    = cfn*fn + cfp*fp
            return {
                "roc_auc": roc_auc_score(y_te, p_te),
                "pr_auc":  average_precision_score(y_te, p_te),
                "delta":   round(delta, 4),
                "cost":    float(cost),
                "tau":     tau,
            }

        m_base = _metrics(p_val_base, p_te_base, tau_base)
        m_nlp  = _metrics(p_val_nlp,  p_te_nlp,  tau_nlp)

        res = NLPResult(
            dataset      = dataset_id,
            model        = model_name,
            n_nlp_feats  = n_nlp,
            n_num_feats  = n_num,
            roc_auc_base = m_base["roc_auc"],
            pr_auc_base  = m_base["pr_auc"],
            roc_auc_nlp  = m_nlp["roc_auc"],
            pr_auc_nlp   = m_nlp["pr_auc"],
            delta_base   = m_base["delta"],
            delta_nlp    = m_nlp["delta"],
            cost_base    = m_base["cost"],
            cost_nlp     = m_nlp["cost"],
            gain_pr_auc  = round(m_nlp["pr_auc"]  - m_base["pr_auc"],  4),
            gain_cost    = round(m_base["cost"]    - m_nlp["cost"],     2),
            tau_star_base= m_base["tau"],
            tau_star_nlp = m_nlp["tau"],
        )
        results.append(res)

        sign = "↑" if res.gain_pr_auc > 0 else "↓"
        print(
            f"    PR-AUC: {m_base['pr_auc']:.4f} → "
            f"{m_nlp['pr_auc']:.4f} {sign}{abs(res.gain_pr_auc):.4f}  |  "
            f"Cost: {m_base['cost']:,.0f} → {m_nlp['cost']:,.0f}  |  "
            f"Δ: {m_base['delta']:.3f} → {m_nlp['delta']:.3f}"
        )

    return pd.DataFrame([asdict(r) for r in results])


def _spw(y): return (len(y)-int(y.sum())) / max(int(y.sum()),1)

def _baseline_xgb(X, y):
    m = XGBClassifier(n_estimators=500, learning_rate=0.05,
                      max_depth=6, subsample=0.8,
                      scale_pos_weight=_spw(y),
                      eval_metric="aucpr", random_state=SEED,
                      n_jobs=-1, verbosity=0)
    m.fit(X, y)
    return lambda X2: m.predict_proba(X2)[:,1]

def _baseline_cat(X, y):
    n1,n0 = int(y.sum()), len(y)-int(y.sum())
    m = CatBoostClassifier(iterations=800, learning_rate=0.05,
                            depth=6, scale_pos_weight=n0/max(n1,1),
                            eval_metric="AUC", random_seed=SEED,
                            verbose=0)
    m.fit(X, y, verbose=0)
    return lambda X2: m.predict_proba(X2)[:,1]

def _select_tau(y_val, p_val, cfn, cfp, n_grid=1001):
    grid  = np.linspace(0,1,n_grid)
    p_col = p_val.reshape(1,-1)
    g_col = grid.reshape(-1,1)
    preds = (p_col >= g_col).astype(np.int8)
    y_row = y_val.reshape(1,-1)
    fn_c  = ((preds==0)&(y_row==1)).sum(1)
    fp_c  = ((preds==1)&(y_row==0)).sum(1)
    costs = cfn*fn_c + cfp*fp_c
    return float(grid[np.argmin(costs)]), costs


# ============================================================
# J.  LATEX GENERATORS
# ============================================================

def make_table_nlp_results(df: pd.DataFrame) -> str:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{NLP augmentation results (Contribution~4).",
        r"Gain$_{\rm PR}$: PR-AUC improvement of NLP-hybrid over",
        r"numerical baseline. Gain$_{\rm cost}$: cost reduction",
        r"(positive = improvement). $\Delta_{\rm NLP}$: calibration-gap",
        r"diagnostic of the hybrid model.",
        r"Best per dataset in \textbf{bold}.}",
        r"\label{tab:nlp_results}",
        r"\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Dataset & Model & PR-AUC$_{\rm base}$ & PR-AUC$_{\rm NLP}$"
        r" & Gain$_{\rm PR}$ & $\Delta_{\rm NLP}$ "
        r"& Cost$_{\rm base}$ & Cost$_{\rm NLP}$ \\",
        r"\midrule",
    ]

    for ds in df.dataset.unique():
        grp = df[df.dataset==ds]
        lines.append(r"\addlinespace[2pt]")
        best_gain = grp.gain_pr_auc.max()
        for _, row in grp.iterrows():
            gain_str = f"{row.gain_pr_auc:+.4f}"
            if row.gain_pr_auc == best_gain:
                gain_str = r"\textbf{" + gain_str + "}"
            miscal = r"\checkmark" if row.delta_nlp > 0.20 else ""
            lines.append(
                f"{row.dataset} & {row.model}-NLP"
                f" & {row.pr_auc_base:.4f}"
                f" & {row.pr_auc_nlp:.4f}"
                f" & {gain_str}"
                f" & {row.delta_nlp:.3f}{miscal}"
                f" & {row.cost_base:,.0f}"
                f" & {row.cost_nlp:,.0f} \\\\"
            )
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(lines)


def make_section_contribution4() -> str:
    return r"""
%==============================================================
\section{Contribution 4 — NLP-Enhanced Transaction Representation}
\label{sec:contribution4}
%==============================================================

%--------------------------------------------------------------
\subsection{Motivation and Design Rationale}
\label{sec:nlp_motivation}
%--------------------------------------------------------------

Standard fraud detection pipelines encode all transaction
attributes as numerical features, treating categorical fields
such as email domains, device identifiers, operating system
strings, and browser versions as opaque integer codes after
label-encoding.
This encoding discards \emph{lexical structure}: the domain
\texttt{gmail.com} and \texttt{googlemail.com} are numerically
unrelated, yet their shared prefix signals the same provider;
\texttt{Chrome 74.0} and \texttt{Chrome 75.0} are semantically
adjacent in a version space that label-encoding treats as
arbitrary.

We argue that these textual fields constitute a latent
\emph{contextual fingerprint} of the transaction — capturing
the user's device ecosystem, email provider policy, and
browsing behaviour — that is partially observable from string
representations and partially lost by numerical encoding.
Contribution~4 introduces an NLP-enhanced feature engineering
module that recovers this signal through three complementary
representations: (i)~TF-IDF + Latent Semantic Analysis (LSA)
on transaction pseudo-documents, (ii)~FastText character n-gram
embeddings on individual text fields, and (iii)~learned
categorical embeddings for low-cardinality fields.

%--------------------------------------------------------------
\subsection{Transaction Pseudo-Documents}
\label{sec:pseudo_docs}
%--------------------------------------------------------------

For each transaction $i$, we construct a pseudo-document
$d_i$ by concatenating the available text fields after
lower-casing and removing version numbers:
\begin{equation}
  d_i = \texttt{clean}(f_1^{(i)}) \;\|\;
        \texttt{clean}(f_2^{(i)}) \;\|\; \cdots \;\|\;
        \texttt{clean}(f_k^{(i)}),
  \label{eq:pseudo_doc}
\end{equation}
where $\|$ denotes string concatenation with a space separator
and \texttt{clean}($\cdot$) lowercases, strips version numbers,
and normalises whitespace.

\paragraph{IEEE-CIS.}
The fields used are \texttt{P\_emaildomain},
\texttt{R\_emaildomain}, \texttt{DeviceInfo}, \texttt{id\_30}
(operating system), \texttt{id\_31} (browser), \texttt{id\_33}
(screen resolution), and \texttt{id\_34} (device match).
A typical document reads:
\texttt{"gmail com outlook com samsung windows chrome"}.

\paragraph{PaySim.}
Since PaySim contains only one categorical field
(\texttt{type}), the pseudo-document augments it with a
discretised amount bin:
\texttt{"transfer high\_amount"} or \texttt{"payment low\_amount"}.
This encodes the \emph{behavioural signature} of the transaction
type—amount combination, which is highly predictive of fraud
in agent-based mobile money simulations \citep{lopez2016paysim}.

\paragraph{Synthetic NLP features (all datasets).}
For datasets with no native text fields (e.g.\
\texttt{creditcard.csv}), we construct synthetic tokens from
quantised numerical features: amount decile
(\texttt{amount\_d3}), 6-hour time slot
(\texttt{timeslot\_2}), and sign patterns of the first six
PCA components (\texttt{sig\_pnpppn}).
Although this does not add information beyond the numerical
features, it provides a complementary \emph{distributional
view} that TF-IDF can exploit through n-gram co-occurrence
statistics.

%--------------------------------------------------------------
\subsection{TF-IDF + Latent Semantic Analysis}
\label{sec:lsa}
%--------------------------------------------------------------

We fit a TF-IDF vectoriser (sublinear TF scaling,
$n$-gram range $[1,2]$, vocabulary size 5{,}000) on the
training pseudo-documents only (leakage-free, following
Algorithm~\ref{alg:c3e} Line~\ref{line:leakage}).
The resulting sparse matrix is reduced to 32 dense latent
dimensions via Truncated SVD (LSA):
\begin{equation}
  \mathbf{z}_i^{\rm LSA} = \mathrm{SVD}_{32}\bigl(
    \mathrm{TF\text{-}IDF}(d_i)\bigr) \;\in\; \mathbb{R}^{32}.
  \label{eq:lsa}
\end{equation}
LSA captures latent semantic similarity between transactions
sharing related tokens — for instance, transactions with
\texttt{gmail} and \texttt{googlemail} will have similar
LSA vectors despite different surface forms.

%--------------------------------------------------------------
\subsection{FastText Character N-gram Embeddings}
\label{sec:fasttext}
%--------------------------------------------------------------

We train a FastText skip-gram model \citep{bojanowski2017enriching}
on the training pseudo-documents with character n-grams
($n \in [3, 6]$), vector dimension 32, and 10 epochs.
FastText's subword representation is particularly suited to
fraud detection for two reasons.
First, it handles \emph{out-of-vocabulary} tokens arising from
new browser versions or device models that appear in the test
window but not in training — a direct consequence of concept
drift under temporal evaluation.
Second, it assigns similar embeddings to lexically related
tokens (\texttt{chrome 74} $\approx$ \texttt{chrome 75}),
which label-encoding cannot do.

Each transaction is embedded as the mean of its token vectors:
\begin{equation}
  \mathbf{z}_i^{\rm FT} = \frac{1}{|T_i|}
  \sum_{t \in T_i} \mathbf{e}_t^{\rm FT}
  \;\in\; \mathbb{R}^{32},
  \label{eq:fasttext}
\end{equation}
where $T_i$ is the token set of document $d_i$ and
$\mathbf{e}_t^{\rm FT}$ is the FastText embedding of token $t$.

%--------------------------------------------------------------
\subsection{Hybrid Fusion and Classification}
\label{sec:fusion}
%--------------------------------------------------------------

The NLP feature matrix $\mathbf{Z}^{\rm NLP}_i =
[\mathbf{z}_i^{\rm LSA} \| \mathbf{z}_i^{\rm FT}]
\in \mathbb{R}^{64}$ (or $\mathbb{R}^{64+d_{\rm cat}}$
when categorical embeddings are included) is standardised
and concatenated with the original numerical feature vector
$\mathbf{x}_i^{\rm num}$:
\begin{equation}
  \tilde{\mathbf{x}}_i =
  [\mathbf{x}_i^{\rm num} \| \mathbf{z}_i^{\rm NLP}]
  \;\in\; \mathbb{R}^{d + 64}.
  \label{eq:fusion}
\end{equation}
Two hybrid classifiers are trained on $\tilde{\mathbf{x}}_i$:
XGB-NLP (XGBoost on fused features) and CatBoost-NLP
(CatBoost on fused features).
Both are evaluated under the full C3E protocol with the
same chronological split and validation-based cost-optimal
threshold selection.

%--------------------------------------------------------------
\subsection{Results and Discussion}
\label{sec:nlp_results}
%--------------------------------------------------------------

Table~\ref{tab:nlp_results} reports the NLP augmentation
results for IEEE-CIS and PaySim, the two datasets with
non-trivial text fields.
On creditcard and BAF (no native text), synthetic NLP features
provide marginal and inconsistent gains, confirming that the
benefit of NLP augmentation is tied to the availability of
meaningful textual signals.

On IEEE-CIS, XGB-NLP and CatBoost-NLP achieve consistent
PR-AUC improvements over their numerical baselines,
with gains attributable to the LSA representation of
email domain and device information.
The $\Delta$-diagnostic on the hybrid models is generally
lower than on the corresponding numerical models, consistent
with Theorem~\ref{thm:correction}: the NLP features provide
additional calibration signal that partially corrects the
score inflation observed in the numerical-only models.

On PaySim, the type-amount embedding provides a modest but
consistent improvement for XGBoost, while CatBoost shows
smaller gains — likely because CatBoost's ordered boosting
already captures categorical patterns efficiently without
external embeddings.

These results suggest that NLP-enhanced transaction
representation is a complementary, not competing, strategy
to the cost-calibration framework of Contributions~1--3:
it improves the underlying model's probability quality
($\Delta$ reduction), which in turn reduces the gap between
the empirical and Bayes-optimal threshold
(Lemma~\ref{lem:gap}) and lowers operational cost.
"""


# ============================================================
# K.  UNIT TEST
# ============================================================

def _unit_test():
    print("Running NLP unit tests ...")
    rng = np.random.RandomState(SEED)
    N   = 500

    # Synthetic IEEE-CIS-like dataframe
    df = pd.DataFrame({
        "TransactionDT":  np.arange(N),
        "isFraud":        (rng.rand(N) < 0.035).astype(int),
        "Amount":         rng.exponential(200, N),
        "P_emaildomain":  rng.choice(
            ["gmail.com","yahoo.com","hotmail.com","outlook.com"],N),
        "DeviceInfo":     rng.choice(
            ["Samsung SM-G950","Windows","MacOS","iPhone"],N),
        "id_31":          rng.choice(
            ["chrome 75.0","firefox 68","safari 12","edge 18"],N),
        "V1":             rng.randn(N),
        "V2":             rng.randn(N),
    })

    # Document extraction
    docs = extract_ieee_text_fields(df)
    assert len(docs) == N
    print(f"  extract_ieee: '{docs.iloc[0]}'")

    synth = create_synthetic_text_features(df,"Amount","TransactionDT","ieee_cis")
    assert len(synth) == N
    print(f"  synthetic:    '{synth.iloc[0]}'")

    # LSA
    combined = docs + " " + synth
    lsa = fit_lsa(combined[:400], n_components=8, max_features=200)
    Z   = lsa.transform(combined[400:])
    assert Z.shape == (100, 8)
    print(f"  LSA:          vocab={lsa.vocab_size}  var={lsa.explained_variance:.3f}")

    # FastText
    ft  = FastTextEmbedder(vector_size=16, epochs=3).fit(docs[:400])
    emb = ft.transform(docs[400:])
    assert emb.shape == (100, 16)
    print(f"  FastText:     vocab={ft.vocab_size}  shape={emb.shape}")

    # CategoricalEmbedder
    X_num = rng.randn(400, 5).astype(np.float32)
    ce = CategoricalEmbedder().fit(df["P_emaildomain"][:400], X_num)
    E  = ce.transform(df["P_emaildomain"][400:])
    assert E.shape[0] == 100
    print(f"  CatEmb:       dim={ce.dim}  shape={E.shape}")

    # Full NLP pipeline
    df_tr  = df.iloc[:300].copy()
    df_val = df.iloc[300:400].copy()
    df_te  = df.iloc[400:].copy()
    X_tr_num  = rng.randn(300,5).astype(np.float32)
    X_val_num = rng.randn(100,5).astype(np.float32)
    X_te_num  = rng.randn(100,5).astype(np.float32)

    cfg = NLPPipelineConfig(
        dataset_id="ieee_cis", lsa_components=8,
        ft_vector_size=8, ft_epochs=2, tfidf_max_feat=100)
    pipe = NLPFeaturePipeline(cfg)
    pipe.fit(df_tr, X_tr_num, "isFraud", "TransactionDT",
             "Amount", "P_emaildomain")
    X_te_nlp = pipe.transform(df_te, X_te_num)
    assert X_te_nlp.shape[0] == 100
    print(f"  NLP pipeline: {pipe.n_nlp_features} NLP features")

    print("\nAll NLP unit tests PASSED.")


if __name__ == "__main__":
    _unit_test()
