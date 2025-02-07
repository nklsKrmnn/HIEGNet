import matplotlib.pyplot as plt
import numpy as np

def visualise_results(data, save_path, colors):
    # Prepare data for visualization
    models = [entry["Model"] for entry in data]
    f1_within = [entry["F1_Within"] for entry in data]
    f1_between = [entry["F1_Between"] for entry in data]
    std_within = [entry["Std_Within"] for entry in data]
    std_between = [entry["Std_Between"] for entry in data]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)

    title_font_size = 20
    label_font_size = 16

    # Plot for "Within Patients" with error bars
    axes[0].bar(models, f1_within, yerr=std_within, color=colors, alpha=0.8, capsize=5)
    axes[0].set_title("F1-Score: Within Patients", fontsize=title_font_size)
    axes[0].set_ylabel("F1 Score", fontsize=title_font_size)
    axes[0].set_ylim(0, 1.0)
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    axes[0].set_xticklabels(models, rotation=30, ha="right", rotation_mode="anchor", fontsize=label_font_size)
    axes[0].tick_params(axis="y", labelsize=label_font_size)

    # Annotate bars for "Within Patients"
    for i, (score, std) in enumerate(zip(f1_within, std_within)):
        axes[0].text(i, score + std + 0.02, f"{score:.2f}\n$\pm$ {std}", ha="center", va="bottom",
                     fontsize=label_font_size)

    # Plot for "Between Patients" with error bars
    axes[1].bar(models, f1_between, yerr=std_between, color=colors, alpha=0.8, capsize=5)
    axes[1].set_title("F1-Score: Between Patients", fontsize=title_font_size)
    axes[1].set_ylim(0, 1.0)
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    axes[1].set_xticklabels(models, rotation=30, ha="right", rotation_mode="anchor", fontsize=label_font_size)

    # Annotate bars for "Between Patients"
    for i, (score, std) in enumerate(zip(f1_between, std_between)):
        axes[1].text(i, score + std + 0.02, f"{score:.2f}\n$\pm$ {std}", ha="center", va="bottom",
                     fontsize=label_font_size)

    # Adjust layout
    plt.tight_layout()

    # Show plot
    # plt.show()

    # Save plot
    plt.savefig(save_path,bbox_inches='tight', pad_inches=0)

# Input data structured as a list of dictionaries
data = [
    {"Model": "Random Forest", "F1_Within": 0.62, "F1_Between": 0.60, "Std_Within": 0.02, "Std_Between": 0.02},
    {"Model": "ResNet-18", "F1_Within": 0.52, "F1_Between": 0.35, "Std_Within": 0.06, "Std_Between": 0.04},
    {"Model": "EfficientNetV2", "F1_Within": 0.32, "F1_Between": 0.53, "Std_Within": 0.02, "Std_Between": 0.07},
    {"Model": "U-Net", "F1_Within": 0.77, "F1_Between": 0.54, "Std_Within": 0.03, "Std_Between": 0.05},
    {"Model": "HIEGNet", "F1_Within": 0.73, "F1_Between": 0.61, "Std_Within": 0.02, "Std_Between": 0.02},
    {"Model": "HIEGNet-JK", "F1_Within": 0.72, "F1_Between": 0.59, "Std_Within": 0.03, "Std_Between": 0.03},
    {"Model": "Hybrid-HIEGNet", "F1_Within": 0.71, "F1_Between": 0.44, "Std_Within": 0.03, "Std_Between": 0.04},
]

colors = plt.cm.tab10(np.linspace(0, 1, 7))  # Generate distinct colors for each model

visualise_results(data, "model_comparison_thesis.pdf", colors)

data_paper = [r for r in data if r["Model"] not in ["Hybrid-HIEGNet", "HIEGNet-JK"]]

visualise_results(data_paper, "model_comparison_paper.pdf", colors[:-2])

# Function to blend color with white
def blend_with_white(color, blend_factor=0.5):
    return color[:3] * (1 - blend_factor) + np.array([1, 1, 1]) * blend_factor

# Reduce saturation for the first 5 colors
new_colors = [blend_with_white(c, 0.5) for c in colors[:5]] + list(colors[5:])

visualise_results(data, "model_comparison_appendix.pdf", new_colors)
