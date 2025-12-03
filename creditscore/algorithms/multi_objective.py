# algorithms/multi_objective.py
def is_dominated(a, b):
    # a,b are tuples
    return all(bi <= ai for ai, bi in zip(a, b)) and any(bi < ai for ai, bi in zip(a, b))

def non_dominated_sort(population):
    """
    population: list of dicts with 'fitness' tuple
    returns flattened list of non-dominated sorted individuals (front-by-front)
    """
    P = population
    if not P:
        return []
    S = [set() for _ in P]
    n = [0 for _ in P]
    fronts = [[]]
    fitnesses = [p["fitness"] for p in P]
    for p in range(len(P)):
        for q in range(len(P)):
            if p == q:
                continue
            if is_dominated(fitnesses[p], fitnesses[q]):
                n[p] += 1
            elif is_dominated(fitnesses[q], fitnesses[p]):
                S[p].add(q)
        if n[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    # flatten fronts into actual individuals
    flattened = []
    for front in fronts:
        for idx in front:
            flattened.append(P[idx])
    return flattened
