# evaluation/visualization.py
import matplotlib.pyplot as plt

def plot_pareto(solutions, title="Pareto front (error vs fairness)"):
    xs = [s["fitness"][0] for s in solutions]
    ys = [s["fitness"][1] for s in solutions]
    plt.figure(figsize=(6,5))
    plt.scatter(xs, ys)
    plt.xlabel("Prediction error (lower better)")
    plt.ylabel("Fairness violation (lower better)")
    plt.title(title)
    plt.grid(True)
    plt.show()
