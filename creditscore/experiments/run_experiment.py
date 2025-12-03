from data.data_loader import load_credit_score_dataset
from data.data_utils import split_among_clients
from clients.financial_institution import FinancialInstitutionClient
from server.federated_server import FederatedServer
from server.communication_layer import encode_fitness_vector
from evaluation.visualization import plot_pareto
from evaluation.report import ExperimentReporter
import numpy as np
import warnings
import time

# Suppress convergence warnings from sklearn
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*ConvergenceWarning.*')


def run_federated_experiment_enhanced(n_clients=3, rounds=4, local_pop_size=12, top_k_local=4,
                                      random_state=42, visualize_final=True, generate_report=True):
    """
    Run federated multi-objective evolutionary algorithm for credit scoring.
    Enhanced version with detailed reporting and monitoring.

    Args:
        n_clients: Number of federated clients
        rounds: Number of federated rounds
        local_pop_size: Local population size per client
        top_k_local: Number of top local solutions to send to server per round
        random_state: Random seed for reproducibility
        visualize_final: Whether to visualize final Pareto front
        generate_report: Whether to generate comprehensive report

    Returns:
        final_global: List of final global Pareto solutions
        report_data: Comprehensive experiment report
    """

    # Initialize reporter
    reporter = ExperimentReporter(output_dir="results")
    round_history = []

    print("\n" + "=" * 70)
    print("  FEDERATED MULTI-OBJECTIVE EVOLUTIONARY CREDIT SCORING (FMO-ECS)")
    print("  Enhanced Experiment with Comprehensive Reporting")
    print("=" * 70 + "\n")

    print("📦 PHASE 1: DATA LOADING & PREPROCESSING")
    print("-" * 70)

    print("Loading dataset...")
    X_train, X_test, y_train, y_test, df, sensitive_attr = load_credit_score_dataset(
        path="data/credit_score.csv"
    )

    print(f"\n📊 Dataset Summary:")
    print(f"   Total samples: {len(df)}")
    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Target distribution (DEFAULT): {y_train.value_counts().to_dict()}")
    print(f"   Sensitive attribute: {sensitive_attr}")

    # FIXED: Extract sensitive attribute values BEFORE splitting
    if sensitive_attr and sensitive_attr in X_train.columns:
        sensitive_attr_values = X_train[sensitive_attr].values.copy()
        print(f"   Sensitive attribute shape: {sensitive_attr_values.shape}")
    else:
        sensitive_attr_values = None
        print(f"   WARNING: Sensitive attribute not found in features!")

    # Split data among clients - BUT WE NEED TO TRACK THE INDICES!
    # IMPORTANT: split_among_clients splits sequentially, so we need to match indices
    clients_data = split_among_clients(X_train, y_train, n_clients=n_clients, random_state=random_state)
    print(f"\n✅ Data split among {n_clients} federated clients")

    print("\n" + "=" * 70)
    print("📦 PHASE 2: CLIENT INITIALIZATION")
    print("-" * 70)

    # Initialize clients
    clients = []
    current_idx = 0  # FIXED: Track current index position

    for i, (Xc, yc) in enumerate(clients_data):
        # FIXED: Extract sensitive attribute for THIS client's data using CURRENT INDEX
        client_size = len(Xc)

        if sensitive_attr_values is not None:
            # Get sensitive attribute for this client's data
            # Use positional indexing: from current_idx to current_idx + client_size
            client_sensitive_attr = sensitive_attr_values[current_idx:current_idx + client_size].copy()
        else:
            client_sensitive_attr = None

        c = FinancialInstitutionClient(
            client_id=i,
            X=Xc,
            y=yc,
            sensitive_attr=client_sensitive_attr,  # FIXED: Now passing correct slice!
            random_state=random_state
        )
        c.initialize_population(pop_size=local_pop_size)
        print(f"\n🏢 Client {i}:")
        print(f"   Samples: {len(Xc)}")
        print(f"   Features: {Xc.shape[1]}")
        print(f"   Population size: {local_pop_size}")
        print(f"   Default rate: {yc.mean():.2%}")
        if client_sensitive_attr is not None:
            print(f"   Sensitive attribute distribution: {np.bincount(client_sensitive_attr.astype(int)).tolist()}")

        clients.append(c)
        current_idx += client_size  # FIXED: Increment index for next client

    server = FederatedServer()
    experiment_config = {
        "n_clients": n_clients,
        "rounds": rounds,
        "local_pop_size": local_pop_size,
        "top_k_local": top_k_local,
        "random_state": random_state,
        "dataset_size": len(df),
        "n_features": X_train.shape[1],
        "sensitive_attr": sensitive_attr
    }

    print("\n" + "=" * 70)
    print("🔄 PHASE 3: FEDERATED EVOLUTIONARY OPTIMIZATION")
    print("-" * 70)

    # Federated learning rounds
    for r in range(rounds):
        round_start = time.time()
        print(f"\n{'=' * 70}")
        print(f"🔴 ROUND {r + 1}/{rounds}")
        print(f"{'=' * 70}")

        client_metrics = []

        for c in clients:
            print(f"\n  🏢 Client {c.client_id}:")
            print(f"     Evaluating population...")

            c.evaluate_population()
            pareto_front = c.local_pareto_front

            local_solutions = c.get_fitness_vectors(top_k=top_k_local)

            # Compute client metrics
            fitnesses = [s["fitness"] for s in pareto_front]

            if fitnesses:
                # Convert fitnesses to numpy array for safe indexing
                fitnesses_array = np.array(fitnesses)
                if fitnesses_array.ndim == 1:
                    fitnesses_array = fitnesses_array.reshape(-1, 1)

                avg_error = float(np.mean(fitnesses_array[:, 0])) if fitnesses_array.shape[1] > 0 else 0.0
                avg_fairness = float(np.mean(fitnesses_array[:, 1])) if fitnesses_array.shape[1] > 1 else 0.0
                avg_regulatory = float(np.mean(fitnesses_array[:, 2])) if fitnesses_array.shape[1] > 2 else 0.0
            else:
                avg_error = 0.0
                avg_fairness = 0.0
                avg_regulatory = 0.0

            client_metric = {
                "client_id": c.client_id,
                "local_pareto_size": len(pareto_front),
                "solutions_sent": len(local_solutions),
                "avg_error": round(avg_error, 4),
                "avg_fairness_violation": round(avg_fairness, 4),
                "avg_regulatory_violation": round(avg_regulatory, 4)
            }
            client_metrics.append(client_metric)

            print(f"     Local Pareto front size: {len(pareto_front)}")
            print(f"     Avg Prediction Error: {avg_error:.4f}")
            print(f"     Avg Fairness Violation: {avg_fairness:.4f}")
            print(f"     Avg Regulatory Violation: {avg_regulatory:.4f}")
            print(f"     Sending {len(local_solutions)} solutions to server")

            encoded = [encode_fitness_vector(s) for s in local_solutions]
            server.receive_from_client(encoded)

        print(f"\n  🖥️  SERVER AGGREGATION:")
        global_solutions = server.aggregate(top_k_out=10, visualize=False)
        print(f"     Aggregated {len(global_solutions)} global Pareto solutions")

        # Log round summary
        round_time = time.time() - round_start
        round_summary = reporter.log_round_summary(
            round_num=r + 1,
            total_rounds=rounds,
            client_metrics=client_metrics,
            global_pareto_size=len(global_solutions),
            aggregation_time=round_time
        )
        round_history.append(round_summary)
        print(f"     Round completion time: {round_time:.2f}s")

        # Distribute global solutions to clients
        print(f"\n  📤 SOLUTION DISTRIBUTION:")
        for c in clients:
            c.integrate_global_solutions(global_solutions)
            print(f"     Client {c.client_id}: Integrated {len(global_solutions)} global solutions")

    print("\n" + "=" * 70)
    print("📦 PHASE 4: FINAL AGGREGATION & ANALYSIS")
    print("-" * 70)

    print("\nCollecting final client solutions...")
    for c in clients:
        c.evaluate_population()
        local_solutions = c.get_fitness_vectors(top_k=top_k_local)
        encoded = [encode_fitness_vector(s) for s in local_solutions]
        server.receive_from_client(encoded)
        print(f"  ✅ Client {c.client_id}: {len(local_solutions)} final solutions")

    final_global = server.aggregate(top_k_out=50, visualize=False)
    print(f"\n🎯 FINAL GLOBAL PARETO FRONT SIZE: {len(final_global)}")

    print("\n" + "=" * 70)
    print("📊 PARETO FRONT ANALYSIS")
    print("-" * 70)

    if final_global:
        pareto_analysis = reporter.generate_pareto_analysis(final_global)

        print(f"\nObjective 1 - Prediction Error (Lower is Better):")
        print(f"  Min:  {pareto_analysis['prediction_error']['min']:.6f}")
        print(f"  Max:  {pareto_analysis['prediction_error']['max']:.6f}")
        print(f"  Mean: {pareto_analysis['prediction_error']['mean']:.6f}")
        print(f"  Std:  {pareto_analysis['prediction_error']['std']:.6f}")

        print(f"\nObjective 2 - Fairness Violation (Lower is Better):")
        print(f"  Min:  {pareto_analysis['fairness_violation']['min']:.6f}")
        print(f"  Max:  {pareto_analysis['fairness_violation']['max']:.6f}")
        print(f"  Mean: {pareto_analysis['fairness_violation']['mean']:.6f}")
        print(f"  Std:  {pareto_analysis['fairness_violation']['std']:.6f}")

        print(f"\nObjective 3 - Regulatory Violation (Lower is Better):")
        print(f"  Min:  {pareto_analysis['regulatory_violation']['min']:.6f}")
        print(f"  Max:  {pareto_analysis['regulatory_violation']['max']:.6f}")
        print(f"  Mean: {pareto_analysis['regulatory_violation']['mean']:.6f}")
        print(f"  Std:  {pareto_analysis['regulatory_violation']['std']:.6f}")

        print(f"\nDiversity Metrics:")
        print(f"  Diversity Score: {pareto_analysis['diversity_metrics']['diversity_score']:.6f}")
        print(f"  Average Pairwise Distance: {pareto_analysis['diversity_metrics']['avg_pairwise_distance']:.6f}")

        print(f"\nTradeoff Analysis:")
        tradeoff = pareto_analysis['tradeoff_analysis']
        print(f"  Accuracy-Fairness Correlation: {tradeoff['accuracy_fairness_correlation']:.4f}")
        print(f"  Accuracy-Regulatory Correlation: {tradeoff['accuracy_regulatory_correlation']:.4f}")
        print(f"  Fairness-Regulatory Correlation: {tradeoff['fairness_regulatory_correlation']:.4f}")
        print(f"  Conflict Level (Higher=Stronger Tradeoffs): {tradeoff['conflict_level']:.4f}")

    print("\n" + "=" * 70)
    print("📈 CONVERGENCE ANALYSIS")
    print("-" * 70)

    conv_report = reporter.generate_convergence_report(round_history)
    print(f"\nRounds: {conv_report['num_rounds']}")
    print(f"Pareto sizes per round: {conv_report['pareto_sizes_per_round']}")
    print(f"Growth: {conv_report['pareto_growth']}")
    print(f"Trend: {conv_report['convergence_trend']}")

    print("\n" + "=" * 70)
    print("🔍 INTERPRETABILITY ANALYSIS")
    print("-" * 70)

    interp_report = reporter.generate_model_interpretability_report(final_global)
    print(f"\nAverage Features Selected: {interp_report['avg_features_selected']:.2f}")
    print(f"Min Features: {interp_report['min_features']}")
    print(f"Max Features: {interp_report['max_features']}")
    print(f"Interpretability Score: {interp_report['interpretability_score']:.4f}")
    print(f"Feature Distribution: {interp_report['feature_usage_distribution']}")
    print(f"Solutions Analyzed: {interp_report['total_solutions_analyzed']}")

    print("\n" + "=" * 70)
    print("⚖️  FAIRNESS AUDIT")
    print("-" * 70)

    audit = reporter.generate_fairness_audit_report(final_global)
    print(f"\nSolutions Audited: {audit['num_solutions_audited']}")
    print(f"Audit Status: {audit['audit_status']}")
    print(f"Fairness Objectives: {', '.join(audit['fairness_objectives'])}")

    # Visualization
    if visualize_final and final_global:
        print("\n📊 Visualizing final Pareto front...")
        plot_pareto(final_global, title="Final Global Pareto Front (FMO-ECS)")

    # Generate comprehensive report
    if generate_report:
        print("\n📄 Generating comprehensive experiment report...")
        report = reporter.save_final_report(final_global, experiment_config, round_history)

    print("\n" + "=" * 70)
    print("✅ EXPERIMENT COMPLETED SUCCESSFULLY")
    print("=" * 70 + "\n")

    return final_global, reporter.experiment_log
