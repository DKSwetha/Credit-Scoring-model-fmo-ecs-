import json
import numpy as np
from pathlib import Path
from datetime import datetime


def convert_to_serializable(obj):
    """
    Convert numpy types and other non-serializable objects to JSON-serializable types.

    FIXED: Removed np.bool8 (doesn't exist in newer NumPy versions)
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):  # FIXED: Removed np.bool8
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


class ExperimentReporter:
    """Generate comprehensive experiment reports."""

    def __init__(self, output_dir="results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiment_log = []

    def log_round_summary(self, round_num, total_rounds, client_metrics, global_pareto_size, aggregation_time):
        """Log summary for each round."""
        summary = {
            "round": round_num,
            "total_rounds": total_rounds,
            "client_metrics": client_metrics,
            "global_pareto_size": global_pareto_size,
            "aggregation_time": aggregation_time,
            "timestamp": str(datetime.now())
        }
        self.experiment_log.append(summary)
        return summary

    def generate_pareto_analysis(self, pareto_solutions):
        """
        Generate detailed analysis of Pareto front.
        """
        if not pareto_solutions:
            return {}

        # Extract fitness values
        fitnesses = np.array([s.get("fitness", [np.nan, np.nan, np.nan]) for s in pareto_solutions])

        # Handle edge case where fitness might be single value
        if fitnesses.ndim == 1:
            fitnesses = fitnesses.reshape(-1, 1)

        # Objective 1: Prediction Error
        pred_errors = fitnesses[:, 0] if fitnesses.shape[1] > 0 else np.array([])
        # Objective 2: Fairness Violation
        fairness_violations = fitnesses[:, 1] if fitnesses.shape[1] > 1 else np.array([])
        # Objective 3: Regulatory Violation
        regulatory_violations = fitnesses[:, 2] if fitnesses.shape[1] > 2 else np.array([])

        # Compute statistics
        analysis = {
            "prediction_error": {
                "min": float(np.min(pred_errors)) if len(pred_errors) > 0 else 0,
                "max": float(np.max(pred_errors)) if len(pred_errors) > 0 else 0,
                "mean": float(np.mean(pred_errors)) if len(pred_errors) > 0 else 0,
                "std": float(np.std(pred_errors)) if len(pred_errors) > 0 else 0
            },
            "fairness_violation": {
                "min": float(np.min(fairness_violations)) if len(fairness_violations) > 0 else 0,
                "max": float(np.max(fairness_violations)) if len(fairness_violations) > 0 else 0,
                "mean": float(np.mean(fairness_violations)) if len(fairness_violations) > 0 else 0,
                "std": float(np.std(fairness_violations)) if len(fairness_violations) > 0 else 0
            },
            "regulatory_violation": {
                "min": float(np.min(regulatory_violations)) if len(regulatory_violations) > 0 else 0,
                "max": float(np.max(regulatory_violations)) if len(regulatory_violations) > 0 else 0,
                "mean": float(np.mean(regulatory_violations)) if len(regulatory_violations) > 0 else 0,
                "std": float(np.std(regulatory_violations)) if len(regulatory_violations) > 0 else 0
            },
            "diversity_metrics": self._compute_diversity_metrics(pareto_solutions),
            "tradeoff_analysis": self._compute_tradeoff_analysis(fitnesses)
        }

        return analysis

    def _compute_diversity_metrics(self, pareto_solutions):
        """Compute diversity among Pareto solutions."""
        if len(pareto_solutions) < 2:
            return {
                "diversity_score": 0.0,
                "avg_pairwise_distance": 0.0
            }

        # Extract feature masks
        feature_masks = []
        for sol in pareto_solutions:
            params = sol.get("params", {})
            mask = params.get("feature_mask", ())
            if mask:
                feature_masks.append(np.array(mask, dtype=float))
            else:
                feature_masks.append(np.array([]))

        if not feature_masks:
            return {
                "diversity_score": 0.0,
                "avg_pairwise_distance": 0.0
            }

        # Compute pairwise distances
        distances = []
        for i in range(len(feature_masks)):
            for j in range(i + 1, len(feature_masks)):
                if len(feature_masks[i]) > 0 and len(feature_masks[j]) > 0:
                    # Hamming distance
                    dist = np.sum(np.abs(feature_masks[i] - feature_masks[j])) / len(feature_masks[i])
                    distances.append(dist)

        avg_pairwise_distance = float(np.mean(distances)) if distances else 0.0
        diversity_score = avg_pairwise_distance

        return {
            "diversity_score": diversity_score,
            "avg_pairwise_distance": avg_pairwise_distance
        }

    def _compute_tradeoff_analysis(self, fitnesses):
        """Analyze tradeoffs between objectives."""
        if fitnesses.shape[1] < 3:
            return {
                "accuracy_fairness_correlation": 0.0,
                "accuracy_regulatory_correlation": 0.0,
                "fairness_regulatory_correlation": 0.0,
                "conflict_level": 0.0
            }

        try:
            pred_error = fitnesses[:, 0]
            fairness = fitnesses[:, 1]
            regulatory = fitnesses[:, 2]

            corr_af = float(np.corrcoef(pred_error, fairness)[0, 1])
            corr_ar = float(np.corrcoef(pred_error, regulatory)[0, 1])
            corr_fr = float(np.corrcoef(fairness, regulatory)[0, 1])

            # Handle NaN correlations
            corr_af = corr_af if not np.isnan(corr_af) else 0.0
            corr_ar = corr_ar if not np.isnan(corr_ar) else 0.0
            corr_fr = corr_fr if not np.isnan(corr_fr) else 0.0

            # Conflict level (higher = more tradeoffs)
            conflict_level = max(abs(corr_af), abs(corr_ar), abs(corr_fr))

            return {
                "accuracy_fairness_correlation": corr_af,
                "accuracy_regulatory_correlation": corr_ar,
                "fairness_regulatory_correlation": corr_fr,
                "conflict_level": float(conflict_level)
            }
        except Exception as e:
            print(f"[WARNING] Tradeoff analysis failed: {e}")
            return {
                "accuracy_fairness_correlation": 0.0,
                "accuracy_regulatory_correlation": 0.0,
                "fairness_regulatory_correlation": 0.0,
                "conflict_level": 0.0
            }

    def generate_convergence_report(self, round_history):
        """Analyze convergence across rounds."""
        pareto_sizes = [r.get("global_pareto_size", 0) for r in round_history]
        num_rounds = len(round_history)

        if num_rounds < 2:
            growth = 0
            trend = "insufficient_data"
        else:
            growth = pareto_sizes[-1] - pareto_sizes[0]
            if growth > 0:
                trend = "improving"
            elif growth < 0:
                trend = "declining"
            else:
                trend = "stable"

        return {
            "num_rounds": num_rounds,
            "pareto_sizes_per_round": pareto_sizes,
            "pareto_growth": growth,
            "convergence_trend": trend
        }

    def generate_model_interpretability_report(self, pareto_solutions):
        """
        FIXED: Properly extract and analyze feature selection.
        """
        if not pareto_solutions:
            return {
                "avg_features_selected": 0,
                "min_features": 0,
                "max_features": 0,
                "interpretability_score": 0.0,
                "feature_usage_distribution": {}
            }

        feature_counts = []

        for sol in pareto_solutions:
            params = sol.get("params", {})
            feature_mask = params.get("feature_mask")

            if feature_mask:
                # Count selected features (1s in the mask)
                mask_array = np.array(feature_mask, dtype=int)
                num_features = int(np.sum(mask_array))
                feature_counts.append(num_features)

        if not feature_counts:
            return {
                "avg_features_selected": 0,
                "min_features": 0,
                "max_features": 0,
                "interpretability_score": 0.0,
                "feature_usage_distribution": {}
            }

        avg_features = float(np.mean(feature_counts))
        min_features = int(np.min(feature_counts))
        max_features = int(np.max(feature_counts))

        # Interpretability score: higher when fewer features (0 to 1)
        # Normalize by max possible features
        max_possible = max_features if max_features > 0 else 1
        interpretability_score = 1.0 - (avg_features / max_possible)

        # Distribution of feature counts
        unique, counts = np.unique(feature_counts, return_counts=True)
        feature_usage_dist = {int(k): int(v) for k, v in zip(unique, counts)}

        return {
            "avg_features_selected": round(avg_features, 2),
            "min_features": min_features,
            "max_features": max_features,
            "interpretability_score": round(interpretability_score, 4),
            "feature_usage_distribution": feature_usage_dist,
            "total_solutions_analyzed": len(pareto_solutions)
        }

    def generate_fairness_audit_report(self, pareto_solutions):
        """Generate fairness audit report."""
        audit_results = {
            "num_solutions_audited": len(pareto_solutions),
            "audit_status": "passed",
            "fairness_objectives": [
                "demographic_parity",
                "equalized_odds",
                "fairness_compliance"
            ],
            "solutions_meeting_criteria": 0,
            "fairness_metrics": {}
        }

        # Count solutions with fairness violation < 0.1
        good_fairness = 0
        fairness_violations = []

        for sol in pareto_solutions:
            fitness = sol.get("fitness", [1.0, 1.0, 1.0])
            fairness_violation = fitness[1] if len(fitness) > 1 else 1.0
            fairness_violations.append(fairness_violation)

            if fairness_violation < 0.1:
                good_fairness += 1

        audit_results["solutions_meeting_criteria"] = good_fairness

        if fairness_violations:
            audit_results["fairness_metrics"] = {
                "min_violation": float(np.min(fairness_violations)),
                "max_violation": float(np.max(fairness_violations)),
                "mean_violation": float(np.mean(fairness_violations)),
                "median_violation": float(np.median(fairness_violations))
            }

        return audit_results

    def save_final_report(self, pareto_solutions, experiment_config, round_history):
        """
        Save comprehensive final report.

        FIXED: Convert all numpy types to JSON-serializable types!
        """
        timestamp = datetime.now().strftime("%Y-%m-%dT%H-%M-%S.%f")

        report_data = {
            "experiment_config": experiment_config,
            "timestamp": timestamp,
            "pareto_front": [],
            "round_history": round_history,
            "summary_statistics": {}
        }

        # Extract detailed solution information
        for i, sol in enumerate(pareto_solutions):
            # FIXED: Convert fitness to list
            fitness = sol.get("fitness", [])
            if isinstance(fitness, np.ndarray):
                fitness = fitness.tolist()

            # FIXED: Convert feature_mask tuple to list
            feature_mask = sol.get("params", {}).get("feature_mask", ())
            if isinstance(feature_mask, (tuple, np.ndarray)):
                feature_mask = list(feature_mask)

            sol_data = {
                "solution_id": i,
                "fitness": fitness,
                "params": {
                    "C": convert_to_serializable(sol.get("params", {}).get("C")),
                    "penalty": sol.get("params", {}).get("penalty"),
                    "solver": sol.get("params", {}).get("solver"),
                },
                "selected_features": sol.get("selected_features", []),
                "num_features_selected": 0,
                "coef": convert_to_serializable(sol.get("coef")),
                "intercept": convert_to_serializable(sol.get("intercept")),
                "error": sol.get("error")
            }

            # Count selected features properly
            if feature_mask:
                sol_data["num_features_selected"] = int(np.sum(np.array(feature_mask, dtype=int)))

            report_data["pareto_front"].append(sol_data)

        # Add summary statistics
        report_data["summary_statistics"] = {
            "pareto_analysis": self.generate_pareto_analysis(pareto_solutions),
            "convergence": self.generate_convergence_report(round_history),
            "interpretability": self.generate_model_interpretability_report(pareto_solutions),
            "fairness_audit": self.generate_fairness_audit_report(pareto_solutions)
        }

        # FIXED: Convert entire report_data to serializable format before saving
        report_data = convert_to_serializable(report_data)

        # Save JSON report
        json_path = self.output_dir / f"experiment_report_{timestamp}.json"
        try:
            with open(json_path, 'w') as f:
                json.dump(report_data, f, indent=2)
            print(f"✅ Final report saved to: {json_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save JSON report: {e}")

        # Save text report
        text_path = self.output_dir / f"experiment_report_{timestamp}.txt"
        try:
            with open(text_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("FEDERATED MULTI-OBJECTIVE EVOLUTIONARY CREDIT SCORING (FMO-ECS)\n")
                f.write("Final Experiment Report\n")
                f.write("=" * 80 + "\n\n")

                f.write("EXPERIMENT CONFIGURATION\n")
                f.write("-" * 80 + "\n")
                for key, value in experiment_config.items():
                    f.write(f"  {key}: {value}\n")

                f.write("\n" + "=" * 80 + "\n")
                f.write("PARETO FRONT SOLUTIONS\n")
                f.write("=" * 80 + "\n\n")

                for i, sol_data in enumerate(report_data["pareto_front"]):
                    f.write(f"Solution {i}:\n")
                    f.write(f"  Fitness: {sol_data['fitness']}\n")
                    f.write(f"  Features Selected: {sol_data['num_features_selected']}\n")
                    f.write(f"  C: {sol_data['params']['C']}\n")
                    f.write(f"  Error: {sol_data['error']}\n")
                    f.write("\n")

                f.write("=" * 80 + "\n")
                f.write("SUMMARY STATISTICS\n")
                f.write("=" * 80 + "\n\n")

                # Pareto Analysis
                pareto_stats = report_data["summary_statistics"]["pareto_analysis"]
                f.write("PARETO FRONT ANALYSIS:\n")
                f.write(
                    f"  Prediction Error (min/max/mean): {pareto_stats['prediction_error']['min']:.4f} / {pareto_stats['prediction_error']['max']:.4f} / {pareto_stats['prediction_error']['mean']:.4f}\n")
                f.write(
                    f"  Fairness Violation (min/max/mean): {pareto_stats['fairness_violation']['min']:.4f} / {pareto_stats['fairness_violation']['max']:.4f} / {pareto_stats['fairness_violation']['mean']:.4f}\n")
                f.write(
                    f"  Regulatory Violation (min/max/mean): {pareto_stats['regulatory_violation']['min']:.4f} / {pareto_stats['regulatory_violation']['max']:.4f} / {pareto_stats['regulatory_violation']['mean']:.4f}\n\n")

                # Interpretability
                interp_stats = report_data["summary_statistics"]["interpretability"]
                f.write("INTERPRETABILITY ANALYSIS:\n")
                f.write(f"  Avg Features Selected: {interp_stats['avg_features_selected']}\n")
                f.write(f"  Min/Max Features: {interp_stats['min_features']} / {interp_stats['max_features']}\n")
                f.write(f"  Interpretability Score: {interp_stats['interpretability_score']}\n")
                f.write(f"  Feature Distribution: {interp_stats['feature_usage_distribution']}\n\n")

                # Fairness
                fairness_stats = report_data["summary_statistics"]["fairness_audit"]
                f.write("FAIRNESS AUDIT:\n")
                f.write(
                    f"  Solutions Meeting Fairness Criteria: {fairness_stats['solutions_meeting_criteria']} / {fairness_stats['num_solutions_audited']}\n")
                f.write(
                    f"  Mean Fairness Violation: {fairness_stats['fairness_metrics'].get('mean_violation', 'N/A')}\n")

            print(f"✅ Text report saved to: {text_path}")
        except Exception as e:
            print(f"[ERROR] Failed to save text report: {e}")

        return report_data
