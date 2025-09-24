
import matplotlib.pyplot as plt

def save_learning_curve(history, path):
    # history: list of dicts with 'step' and 'loss'
    steps = [h["step"] for h in history]
    losses = [h["loss"] for h in history]
    plt.figure()
    # plot learning curve as a continuous line (no markers)
    plt.plot(steps, losses)
    plt.xlabel("step")
    plt.ylabel("loss")
    plt.yscale('log')
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
