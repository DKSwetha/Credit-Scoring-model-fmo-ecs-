import json
import zlib
import base64
import numpy as np


def _to_python(obj):
    """
    Converts numpy types to pure Python so JSON can serialize.
    Works recursively for lists, tuples, dicts.
    """
    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, (list, tuple)):
        return [_to_python(o) for o in obj]

    if isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}

    return obj  # already a Python type


def encode_fitness_vector(solution, compress=True):
    """
    Compress + encode solution dict (fitness + params summary).
    Converts numpy values to Python types to avoid JSON errors.

    FIXED: Properly handles numpy arrays by converting to list.
    """
    params = solution.get("params", {})

    # Convert params to Python-compatible types
    params_py = _to_python(params)

    # Reconstruct feature_mask as list (since JSON doesn't support tuples)
    # but preserve the intent that it should be a tuple on decode
    if isinstance(params_py, dict) and "feature_mask" in params_py:
        params_py["feature_mask"] = list(params_py.get("feature_mask", []))

    # CRITICAL FIX: Convert fitness properly (numpy array → list)
    fitness = solution.get("fitness", [])
    if isinstance(fitness, np.ndarray):
        fitness = fitness.tolist()
    elif isinstance(fitness, (list, tuple)):
        fitness = list(fitness)
    else:
        fitness = []

    payload = {
        "fitness": fitness,
        "params": params_py
    }

    raw = json.dumps(payload).encode("utf-8")

    if compress:
        raw = zlib.compress(raw)

    return base64.b64encode(raw).decode("ascii")


def decode_fitness_vector(encoded, compress=True):
    """
    Decode fitness vector. Converts lists back to numpy arrays.

    FIXED: Reconstructs numpy arrays from lists for fitness.
    """
    try:
        raw = base64.b64decode(encoded.encode("ascii"))
        if compress:
            raw = zlib.decompress(raw)
        decoded = json.loads(raw)

        # Reconstruct fitness as numpy array
        if isinstance(decoded.get("fitness"), list):
            decoded["fitness"] = np.array(decoded["fitness"], dtype=np.float64)

        # Reconstruct feature_mask tuple
        if isinstance(decoded.get("params"), dict) and isinstance(decoded["params"].get("feature_mask"), list):
            decoded["params"]["feature_mask"] = tuple(decoded["params"]["feature_mask"])

        return decoded
    except Exception as e:
        print(f"[communication_layer] Error decoding fitness vector: {e}")
        raise
