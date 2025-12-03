from experiments.run_experiment import run_federated_experiment_enhanced

if __name__ == "__main__":
    """
    Federated Multi-Objective Evolutionary Credit Scoring (FMO-ECS)

    This system implements a federated learning approach to credit scoring
    that optimizes for multiple objectives:
    1. Prediction accuracy (minimize error)
    2. Fairness (minimize demographic parity violations)
    3. Regulatory compliance (minimize constraint violations)

    Architecture Components:
    - Client-side: Local data, feature engineering, evolutionary optimization
    - Server-side: Secure aggregation, NSGA-II optimization, compliance monitoring
    - Communication: Encrypted fitness vectors, privacy-preserving aggregation
    """

    print("\n" + "=" * 80)
    print(" FEDERATED MULTI-OBJECTIVE EVOLUTIONARY CREDIT SCORING SYSTEM (FMO-ECS)")
    print(" Research Paper Implementation")
    print("=" * 80 + "\n")

    # Run enhanced experiment with comprehensive reporting
    final_solutions, experiment_log = run_federated_experiment_enhanced(
        n_clients=3,  # Number of federated institutions
        rounds=3,  # Number of federated rounds
        local_pop_size=12,  # Local population size per client
        top_k_local=4,  # Top solutions to send per round
        random_state=42,  # Reproducibility
        visualize_final=True,  # Visualize Pareto front
        generate_report=True  # Generate comprehensive report
    )

    print("\n" + "=" * 80)
    print(" RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n✅ Final Global Pareto Front Size: {len(final_solutions)}")
    print(f"✅ Experiment Log Entries: {len(experiment_log)}")
    print(f"\n📁 Detailed reports saved to: results/ directory")
    print(f"   - experiment_report_*.json (machine-readable)")
    print(f"   - experiment_report_*.txt (human-readable)")
    print("\n" + "=" * 80 + "\n")

