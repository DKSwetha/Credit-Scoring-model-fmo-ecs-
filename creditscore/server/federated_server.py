from algorithms.multi_objective import non_dominated_sort
from evaluation.visualization import plot_pareto
from server.communication_layer import decode_fitness_vector


class FederatedServer:
    def __init__(self):
        # store decoded solution dicts here
        self.received = []

    def receive_from_client(self, solutions):
        """
        `solutions` may be either:
         - a list of dicts (already decoded), or
         - a list of encoded strings (base64 compressed JSON) produced by encode_fitness_vector.
        This method decodes strings and appends dicts to self.received.
        """
        for s in solutions:
            # if s is a string (encoded), decode it
            if isinstance(s, str):
                try:
                    decoded = decode_fitness_vector(s)
                    # decoded is a dict with keys 'fitness' and 'params'
                    # Make sure types are consistent (e.g., tuples/lists)
                    # Convert 'fitness' back to tuple if it was list (handled in decode_fitness_vector)
                    if isinstance(decoded.get("fitness"), list):
                        decoded["fitness"] = tuple(decoded["fitness"])

                    # FIXED: Preserve all metadata fields from decoded solution
                    entry = {
                        "params": decoded.get("params"),
                        "fitness": decoded.get("fitness"),
                        "model_blob": decoded.get("model_blob"),
                        "coef": decoded.get("coef"),
                        "intercept": decoded.get("intercept"),
                        "selected_features": decoded.get("selected_features"),
                        "error": decoded.get("error")
                    }
                    self.received.append(entry)
                except Exception as e:
                    # if decoding fails, log and skip the entry
                    print(f"[Server] failed to decode payload: {e}")
                    continue
            elif isinstance(s, dict):
                # already a dict: append as-is
                self.received.append(s)
            else:
                # unexpected type: try to ignore or log
                print(f"[Server] unexpected payload type: {type(s)}")
                continue

    def aggregate(self, top_k_out=10, visualize=False):
        """
        Aggregate received solutions using non-dominated sorting.
        Returns top_k solutions from the Pareto front.
        """
        # only consider items that have fitness
        population = [s for s in self.received if s.get("fitness") is not None]
        if not population:
            print("[Server] No valid solutions to aggregate")
            return []

        sorted_pop = non_dominated_sort(population)
        aggregated = sorted_pop[:top_k_out]

        if visualize:
            plot_pareto(aggregated, title="Global Pareto front")

        # Clear buffer for next round (keeps previous behavior)
        self.received = []
        return aggregated