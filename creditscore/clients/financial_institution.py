import numpy as np
import pandas as pd
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from evaluation.metrics import compute_fitness

# Suppress convergence warnings from sklearn
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')


class FinancialInstitutionClient:
    def __init__(self, client_id, X, y,
                 sensitive_attr=None,
                 regulatory_config=None,
                 random_state=0):

        self.client_id = client_id
        self.X = X.copy().reset_index(drop=True)
        self.y = y.copy().reset_index(drop=True)

        # Fairness target (sensitive attribute for fairness evaluation)
        # FIXED: Ensure this is stored properly and aligned with X/y
        if sensitive_attr is not None:
            sensitive_attr_arr = np.asarray(sensitive_attr)
            # Ensure it matches the size of X
            if len(sensitive_attr_arr) == len(self.X):
                self.sensitive_attr = sensitive_attr_arr.copy()
            else:
                print(
                    f"[WARNING] Client {client_id}: sensitive_attr size {len(sensitive_attr_arr)} != X size {len(self.X)}")
                self.sensitive_attr = None
        else:
            self.sensitive_attr = None

        # Regulatory constraints
        self.regulatory_config = regulatory_config

        self.random_state = random_state
        self.local_population = []
        self.local_pareto_front = []

    def initialize_population(self, pop_size=20):
        """Initialize population with random feature masks and hyperparameters."""
        rng = np.random.RandomState(self.random_state + self.client_id)
        n_features = self.X.shape[1]

        pop = []
        for i in range(pop_size):
            # Feature mask (feature selection bit vector)
            mask = tuple(rng.choice([0, 1], size=n_features, p=[0.3, 0.7]))

            params = {
                "C": float(10 ** (rng.uniform(-2, 2))),  # inverse regularization
                "penalty": "l2",
                "solver": "saga",
                "feature_mask": mask,
                "random_state": int(self.random_state + i + self.client_id),
            }
            pop.append({"params": params, "fitness": None})

        self.local_population = pop

    def evaluate_candidate(self, candidate):
        """
        Train Logistic Regression on masked + encoded + scaled features with a train/test split.
        - Split by indices so masks map correctly to rows.
        - Train on train set, evaluate on test set (compute_fitness uses test DataFrame).
        - Saves coef/intercept, selected_features and logs errors.

        FIXED: Now properly handles sensitive_attr indices!
        """
        import traceback

        params = candidate["params"]

        # 1) Apply feature mask (DataFrame to keep column names)
        if "feature_mask" in params:
            mask_arr = np.array(params["feature_mask"], dtype=bool)
            if mask_arr.sum() == 0:
                fallback = np.array([1.0, 1.0, 1.0])
                candidate.update({
                    "fitness": fallback, "model_blob": None, "coef": None,
                    "selected_features": [], "error": "empty_feature_mask"
                })
                return fallback
            X_used_df = self.X.loc[:, mask_arr].copy()
            selected_features = list(X_used_df.columns)
        else:
            X_used_df = self.X.copy()
            selected_features = list(X_used_df.columns)

        # ensure indices align
        n_samples = len(self.y)
        indices = np.arange(n_samples)

        # 2) Create a small holdout split for evaluation (30% test)
        try:
            train_idx, test_idx = train_test_split(
                indices, test_size=0.3, random_state=params.get("random_state", self.random_state), stratify=self.y
            )
        except Exception:
            # fallback if stratify fails (e.g., small groups)
            train_idx, test_idx = train_test_split(
                indices, test_size=0.3, random_state=params.get("random_state", self.random_state)
            )

        # Build train/test DataFrames for original (unencoded) data (for compute_fitness)
        X_train_df = X_used_df.iloc[train_idx].reset_index(drop=True)
        X_test_df = X_used_df.iloc[test_idx].reset_index(drop=True)
        y_train = self.y.iloc[train_idx].reset_index(drop=True)
        y_test = self.y.iloc[test_idx].reset_index(drop=True)

        # FIXED: Extract sensitive attribute for TEST indices only!
        # CRITICAL: Use iloc to access by position, not by index value
        if self.sensitive_attr is not None and len(self.sensitive_attr) == n_samples:
            try:
                # Create a Series with the sensitive_attr and use iloc with test_idx
                sensitive_attr_series = pd.Series(self.sensitive_attr)
                sensitive_attr_test = sensitive_attr_series.iloc[test_idx].values
            except Exception as e:
                print(f"[WARNING] Failed to extract sensitive_attr for test indices: {e}")
                sensitive_attr_test = None
        else:
            sensitive_attr_test = None

        # 3) Encode categorical features (get_dummies) separately for train/test but ensure same columns
        try:
            X_train_enc = pd.get_dummies(X_train_df, drop_first=True)
            X_test_enc = pd.get_dummies(X_test_df, drop_first=True)
            # Align columns (fill missing with zeros)
            X_test_enc = X_test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
        except Exception as e:
            fallback = np.array([1.0, 1.0, 1.0])
            candidate.update({
                "fitness": fallback, "model_blob": None, "coef": None,
                "selected_features": selected_features,
                "error": f"encoding_failed: {str(e)}"
            })
            Path("results/logs").mkdir(parents=True, exist_ok=True)
            with open(Path("results/logs") / "failures.log", "a", encoding="utf-8") as lf:
                lf.write(f"Encoding failed client {self.client_id} params={params}\n{traceback.format_exc()}\n\n")
            return fallback

        # 4) Check for zero columns after encoding
        if X_train_enc.shape[1] == 0:
            fallback = np.array([1.0, 1.0, 1.0])
            candidate.update({
                "fitness": fallback, "model_blob": None, "coef": None,
                "selected_features": selected_features,
                "error": "no_columns_after_encoding"
            })
            return fallback

        # 5) Scale features (fit on train, transform both)
        scaler = StandardScaler()
        try:
            X_train_scaled = scaler.fit_transform(X_train_enc)
            X_test_scaled = scaler.transform(X_test_enc)
        except Exception as e:
            fallback = np.array([1.0, 1.0, 1.0])
            candidate.update({
                "fitness": fallback, "model_blob": None, "coef": None,
                "selected_features": selected_features,
                "error": f"scaling_failed: {str(e)}"
            })
            Path("results/logs").mkdir(parents=True, exist_ok=True)
            with open(Path("results/logs") / "failures.log", "a", encoding="utf-8") as lf:
                lf.write(f"Scaling failed client {self.client_id} params={params}\n{traceback.format_exc()}\n\n")
            return fallback

        # 6) Train logistic model on train set
        clf = LogisticRegression(
            C=params.get("C", 1.0),
            penalty=params.get("penalty", "l2"),
            solver=params.get("solver", "saga"),
            max_iter=2000,
            tol=1e-4,
            random_state=params.get("random_state", 42),
            n_jobs=-1
        )

        try:
            clf.fit(X_train_scaled, y_train)
            y_pred_test = clf.predict(X_test_scaled)

            # 7) Evaluate on test set - CRITICAL FIX: Pass only test sensitive_attr!
            # compute_fitness(y_true, y_pred, sensitive_attr=None)
            fitness = compute_fitness(
                y_test,  # ← y_true (test labels)
                y_pred_test,  # ← y_pred (test predictions)
                sensitive_attr=sensitive_attr_test  # ← FIXED: test indices only!
            )

            candidate["model_blob"] = None
            candidate["fitness"] = fitness
            try:
                candidate["coef"] = clf.coef_.tolist()
                candidate["intercept"] = clf.intercept_.tolist()
            except Exception:
                candidate["coef"] = None
                candidate["intercept"] = None
            candidate["selected_features"] = selected_features
            candidate["error"] = None

            return fitness

        except Exception as e:
            fallback = np.array([1.0, 1.0, 1.0])
            candidate.update({
                "fitness": fallback, "model_blob": None, "coef": None,
                "selected_features": selected_features,
                "error": f"train_failed: {str(e)}"
            })
            Path("results/logs").mkdir(parents=True, exist_ok=True)
            with open(Path("results/logs") / "failures.log", "a", encoding="utf-8") as lf:
                lf.write(f"Training failed client {self.client_id} params={params}\n{traceback.format_exc()}\n\n")
            return fallback

    def evaluate_population(self):
        """Evaluate all individuals in the population."""
        for cand in self.local_population:
            if cand.get("fitness") is None:
                self.evaluate_candidate(cand)

        # Fallback safety - ensure all have fitness as numpy array
        for cand in self.local_population:
            if cand.get("fitness") is None:
                cand["fitness"] = np.array([1.0, 1.0, 1.0])
            # Convert to numpy array if it's a tuple or list
            if not isinstance(cand["fitness"], np.ndarray):
                cand["fitness"] = np.array(cand["fitness"])

        # Local Pareto approximation: sort by sum of objectives
        self.local_pareto_front = sorted(
            self.local_population,
            key=lambda c: sum(c["fitness"])
        )
        return self.local_pareto_front

    def get_fitness_vectors(self, top_k=5):
        """Get top-k solutions from local Pareto front."""
        out = []
        for c in self.local_pareto_front[:top_k]:
            out.append({
                "params": c["params"],
                "fitness": c["fitness"],
                "model_blob": c.get("model_blob"),
                "coef": c.get("coef"),
                "intercept": c.get("intercept"),
                "selected_features": c.get("selected_features"),
                "error": c.get("error")
            })
        return out

    def integrate_global_solutions(self, global_solutions):
        """Integrate global solutions from server into local population."""
        combined = (
                self.local_population +
                [{"params": s["params"], "fitness": None} for s in global_solutions]
        )
        # keep same population size
        self.local_population = combined[:len(self.local_population)]
