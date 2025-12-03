# algorithms/evolutionary.py
import random
import copy

def tournament_selection(pop, k=3):
    """
    Tournament selection based on fitness tuple.
    Lower tuple = better (error, fairness, regulatory).
    """
    candidates = random.sample(pop, min(k, len(pop)))
    return min(candidates, key=lambda c: sum(c["fitness"]))


def sbx_crossover(parent1, parent2):
    """
    Crossover for Logistic Regression hyperparameters + feature masks.

    Parameters crossed:
    - C (inverse regularization strength)
    - feature_mask (bitwise crossover)
    """

    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)

    p1 = parent1["params"]
    p2 = parent2["params"]

    # -----------------------------
    # 1️⃣ CROSSOVER: REGULARIZATION C
    # -----------------------------
    # Use geometric mean crossover for numeric hyperparameters
    c1 = (p1["C"] * p2["C"]) ** 0.5
    c2 = c1 * random.uniform(0.9, 1.1)

    child1["params"]["C"] = c1
    child2["params"]["C"] = max(1e-4, c2)

    # -----------------------------
    # 2️⃣ CROSSOVER: FEATURE MASKS
    # -----------------------------
    if "feature_mask" in p1 and "feature_mask" in p2:
        m1 = list(p1["feature_mask"])
        m2 = list(p2["feature_mask"])
        n = len(m1)

        if n > 1:
            cut = random.randint(1, n - 1)
        else:
            cut = 1

        child1_mask = tuple(m1[:cut] + m2[cut:])
        child2_mask = tuple(m2[:cut] + m1[cut:])

        child1["params"]["feature_mask"] = child1_mask
        child2["params"]["feature_mask"] = child2_mask

    # Reset fitness
    child1["fitness"] = None
    child2["fitness"] = None

    return child1, child2


def polynomial_mutation(individual, indpb=0.2):
    """
    Mutation operator:
    - Mutates C via log-scale perturbation
    - Bit-flip mutation for feature_mask
    """

    params = individual["params"]

    # -----------------------------
    # 1️⃣ MUTATE C (inverse regularization)
    # -----------------------------
    if random.random() < indpb:
        factor = 10 ** random.uniform(-0.3, 0.3)
        params["C"] = max(1e-4, params["C"] * factor)

    # -----------------------------
    # 2️⃣ MUTATE FEATURE MASK (bit flips)
    # -----------------------------
    if "feature_mask" in params:
        mask = list(params["feature_mask"])
        for i in range(len(mask)):
            # small bit-flip probability
            if random.random() < 0.05:
                mask[i] = 1 - mask[i]
        params["feature_mask"] = tuple(mask)

    # Reset fitness since model changed
    individual["fitness"] = None
    return individual
