import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def _map_to_binary(series):
    """Map a 2-valued series to 0/1 robustly."""
    s = series.copy()
    if pd.api.types.is_numeric_dtype(s):
        vals = set(s.dropna().unique())
        if vals.issubset({0, 1}):
            return s.astype(int)
        # if values are like {1,2} map max->1
        if len(vals) == 2:
            try:
                nums = sorted([float(v) for v in vals])
                mapping = {nums[0]: 0, nums[1]: 1}
                return s.map(lambda x: mapping.get(float(x), np.nan)).astype(int)
            except Exception:
                pass
    # non-numeric with two values
    unique_vals = list(pd.Series(s).dropna().unique())
    if len(unique_vals) == 2:
        lowered = [str(v).strip().lower() for v in unique_vals]
        # try common positive tokens
        if any(tok in lowered for tok in ("yes", "y", "true", "t", "1", "approved", "good")):
            mapping = {}
            for v in unique_vals:
                mapping[v] = 1 if str(v).strip().lower() in ("yes", "y", "true", "t", "1", "approved", "good") else 0
            return s.map(mapping).astype(int)
        # fallback: first->0 second->1
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        return s.map(mapping).astype(int)
    raise ValueError("Cannot map series to binary. Provide a binary target column (0/1).")


def load_credit_score_dataset(path="data/credit_score.csv",
                              target_col="DEFAULT",
                              sensitive_col_preferred=None,
                              test_size=0.2,
                              random_state=42):
    """
    Load and preprocess the credit_score.csv dataset.

    FIXED: Now uses CAT_DEPENDENTS as sensitive attribute (proxy for family status).
    Returns sensitive_attr as binary array (0=no dependents, 1=has dependents).

    Returns:
        X_train, X_test, y_train, y_test, original_df, sensitive_attr_name (or None)
    """
    # 1. Resolve path - check multiple locations
    possible_paths = [
        path,
        os.path.join(os.getcwd(), path),
        os.path.join(os.path.dirname(__file__), "..", path),
        os.path.abspath(path)
    ]

    resolved_path = None
    for p in possible_paths:
        if os.path.exists(p):
            resolved_path = p
            break

    if resolved_path is None:
        print(f"\n[ERROR] Could not find credit_score.csv!")
        print(f"Looked in these locations:")
        for p in possible_paths:
            print(f"  - {os.path.abspath(p)}")
        print(f"\nCurrent working directory: {os.getcwd()}")
        print(f"\nPlease ensure your CSV file is in one of these locations.")
        raise FileNotFoundError(f"Dataset not found at: {path}")

    # 2. load
    df = pd.read_csv(resolved_path)
    df.columns = [c.strip() for c in df.columns]

    print(f"[data_loader] Loaded {resolved_path} shape={df.shape}")
    print(f"[data_loader] Columns: {list(df.columns)}")

    # 3. drop ID column if present
    if "CUST_ID" in df.columns:
        print("[data_loader] Dropping identifier column 'CUST_ID'")
        df = df.drop(columns=["CUST_ID"], errors="ignore")

    # 4. ensure target exists and map to binary
    if target_col not in df.columns:
        # Try to find a likely target column
        print(f"[data_loader] WARNING: target_col '{target_col}' not found in columns: {list(df.columns)}")
        print(f"[data_loader] Available columns: {list(df.columns)}")
        raise ValueError(f"[data_loader] target_col '{target_col}' not found. Columns: {list(df.columns)}")

    try:
        y = _map_to_binary(df[target_col])
    except Exception as e:
        raise ValueError(f"[data_loader] Failed to map target '{target_col}' to binary: {e}")

    # 5. FIXED: Detect sensitive attribute - prefer CAT_DEPENDENTS
    sensitive_attr = None
    if sensitive_col_preferred and sensitive_col_preferred in df.columns:
        sensitive_attr = sensitive_col_preferred
        print(f"[data_loader] Using provided sensitive attribute: '{sensitive_attr}'")
    elif 'CAT_DEPENDENTS' in df.columns:
        # Use CAT_DEPENDENTS as the sensitive attribute (family status proxy)
        sensitive_attr = 'CAT_DEPENDENTS'
        print(f"[data_loader] Using sensitive attribute: '{sensitive_attr}' (family status proxy)")
    else:
        # Fallback to other common attributes
        for cand in ['sex', 'gender', 'Sex', 'Gender', 'age_group', 'AGE_GROUP', 'Age', 'age']:
            if cand in df.columns:
                sensitive_attr = cand
                print(f"[data_loader] Detected sensitive attribute: '{sensitive_attr}'")
                break
        if sensitive_attr is None:
            print("[data_loader] WARNING: No sensitive attribute detected")

    # 6. build features X (drop target)
    X = df.drop(columns=[target_col], errors="ignore")

    # drop index-like columns
    unnamed = [c for c in X.columns if str(c).lower().startswith("unnamed")]
    if unnamed:
        print(f"[data_loader] Dropping unnamed/index-like columns: {unnamed}")
        X = X.drop(columns=unnamed, errors="ignore")

    # 7. drop any constant columns (zero variance)
    nunique = X.nunique(dropna=False)
    const_cols = list(nunique[nunique <= 1].index)
    if const_cols:
        print(f"[data_loader] Dropping constant columns: {const_cols}")
        X = X.drop(columns=const_cols, errors="ignore")

    # 8. prepare numeric and categorical lists
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # if sensitive is categorical and present, don't one-hot it
    if sensitive_attr is not None and sensitive_attr in cat_cols:
        cat_cols.remove(sensitive_attr)

    # 9. fill missing values
    for c in num_cols:
        med = X[c].median() if not X[c].isna().all() else 0
        X[c] = X[c].fillna(med)
    for c in cat_cols:
        X[c] = X[c].fillna('missing')

    # 10. one-hot encode categorical columns (excluding sensitive)
    if cat_cols:
        X = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # 11. FIXED: Ensure sensitive attr present in X as binary (0/1)
    if sensitive_attr is not None and sensitive_attr in df.columns:
        if sensitive_attr not in X.columns:
            # Extract and binarize sensitive attribute
            sens_raw = df[sensitive_attr].copy()
            # Convert to binary: 0 if no dependents, 1 if has dependents
            if pd.api.types.is_numeric_dtype(sens_raw):
                X[sensitive_attr] = (sens_raw > 0).astype(int)
            else:
                # For categorical, map to binary
                X[sensitive_attr] = (sens_raw != 0).astype(int)
        else:
            # Already in X, ensure it's binary
            X[sensitive_attr] = (X[sensitive_attr] > 0).astype(int)

        print(f"[data_loader] Sensitive attribute '{sensitive_attr}' binarized (0=no, 1=yes)")
        print(f"[data_loader] Sensitive attribute distribution: {X[sensitive_attr].value_counts().to_dict()}")

    # 12. final checks and prints
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    print(f"[data_loader] Final X shape: {X.shape}, y length: {len(y)}")
    print(f"[data_loader] Feature preview (first 10): {list(X.columns[:10])}")
    if sensitive_attr:
        print(f"[data_loader] Sensitive attribute preserved: '{sensitive_attr}' (binary)")

    # 13. train-test split (stratify if possible)
    stratify = y if len(y.unique()) > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )

    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), \
        y_train.reset_index(drop=True), y_test.reset_index(drop=True), df, sensitive_attr
