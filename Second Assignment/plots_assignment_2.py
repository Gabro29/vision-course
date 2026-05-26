
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.lines import Line2D


def compare_descriptors(folder):

    data_to_plots = []

    for name_dir in folder:
        current_counts = []
        all_descriptors_dir = os.path.join(os.getcwd(), "checkpoints", name_dir, "bovw", "descriptors")

        if os.path.exists(all_descriptors_dir):
            for filename in os.listdir(all_descriptors_dir):
                if filename.endswith('.npy'):
                    file_path = os.path.join(all_descriptors_dir, filename)
                    data = np.load(file_path, mmap_mode='r')
                    current_counts.append(data.shape[0])

        data_to_plots.append(current_counts)

    plt.figure(figsize=(8, 8))
    box = plt.boxplot(data_to_plots,
                      vert=True,
                      patch_artist=True,
                      tick_labels=["gray", "pca"],
                      widths=0.7,
                      medianprops=dict(color="black", linewidth=2),
                      showmeans=True,
                      meanprops=dict(marker='o', markerfacecolor='white', markeredgecolor='black', markersize=8))

    colors = ['gray', '#3498db', '#2ecc71']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    plt.ylabel("Numero di descrittori", fontsize=12)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=12))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


def compare_f1_scores(folder):
    plt.figure(figsize=(8, 8))
    colors = ['#e74c3c', 'gray']
    labels = ["pca", "gray"]

    for idx, name_dir in enumerate(folder):
        csv_path = os.path.join(os.getcwd(), "checkpoints", name_dir, "classification", "model_summary.csv")
        df = pd.read_csv(csv_path)

        df = df.sort_values(by='K')
        df = df[df["model"] == "Soft"]
        plt.plot(df['K'], df['f1_macro'],
                 marker='o',
                 linewidth=2,
                 markersize=8,
                 color=colors[idx % len(colors)],
                 label=labels[idx])

    plt.xlabel("Dimensione del vocabolario", fontsize=12)
    plt.ylabel("F1 macro", fontsize=12)

    plt.legend(fontsize=12, loc="best")

    plt.xticks([50, 100, 500])
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=18))
    plt.grid(axis='both', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def extract_features_count(folder_name):
    raw_num = folder_name.split("_")[1]
    raw_mul = folder_name.split("_")[2]
    raff_mul = 1
    if raw_mul == "mln":
        raff_mul = 1_000_000
    elif raw_mul == "k":
        raff_mul = 1_000
    how_many = int(raw_num) * raff_mul
    return how_many


def compare_f1_with_variable_markers(folders):
    plt.figure(figsize=(10, 8))

    line_colors = {
        "gray": "gray",
        "pca": "#3498db"
    }

    feature_colors = {
        100_000: "#e74c3c",
        1_000_000: "#2ecc71",
        2_000_000: "blue",
        8_000_000: "purple"
    }

    for name_dir in folders:
        csv_path = os.path.join(os.getcwd(), "checkpoints", name_dir, "classification", "model_summary.csv")

        df = pd.read_csv(csv_path)
        df = df.sort_values(by='K')
        df = df[df["model"] == "Soft"]

        mode = "pca" if "pca" in name_dir.lower() else "gray"
        line_color = line_colors[mode]

        num_features = extract_features_count(name_dir)
        marker_color = feature_colors.get(num_features, "black")

        base_size = 8
        size_multiplier = (num_features / 100_000) * 0.5
        dynamic_marker_size = min(base_size + size_multiplier, 25)

        plt.plot(df['K'], df['f1_macro'],
                 linewidth=2,
                 color=line_color,
                 marker='o',
                 markerfacecolor=marker_color,
                 markeredgecolor='black',
                 markeredgewidth=1.2,
                 markersize=dynamic_marker_size)

    plt.xlabel("Dimensione del vocabolario", fontsize=12)
    plt.ylabel("F1 macro", fontsize=12)

    legend_elements = [
        # Line2D([0], [0], color='none', label=r'$\bf{Pre-processing\ (Linee)}$'),
        Line2D([0], [0], color=line_colors['pca'], lw=3, label='PCA'),
        Line2D([0], [0], color=line_colors['gray'], lw=3, label='Gray'),

        Line2D([0], [0], color='none', label=''),

        # Sezione 2: Significato dei Cerchi
        # Line2D([0], [0], color='none', label=r'$\bf{Numero\ di\ Features\ (Cerchi)}$'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=feature_colors[100_000], markeredgecolor='black',
               markersize=10, label='100K SIFT'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=feature_colors[1_000_000], markeredgecolor='black',
               markersize=12, label='1M SIFT'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=feature_colors[2_000_000], markeredgecolor='black',
               markersize=12, label='2.5M SIFT'),
        # Line2D([0], [0], marker='o', color='w', markerfacecolor=feature_colors[8_000_000], markeredgecolor='black',
        #        markersize=14, label='8M SIFT')
    ]

    plt.legend(handles=legend_elements, loc="best", fontsize=11, framealpha=0.9)

    plt.xticks([50, 100, 500])
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=18))
    plt.grid(axis='both', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


def compare_best_classifiers(folders):
    plt.figure(figsize=(10, 8))

    colors = {
        "pca": "#3498db",  # Blu
        "gray": "gray"  # Grigio
    }

    for name_dir in folders:
        csv_path = os.path.join(os.getcwd(), "checkpoints", name_dir, "classification", "model_summary.csv")

        df = pd.read_csv(csv_path)

        mode = "pca" if "pca" in name_dir.lower() else "gray"
        base_color = colors[mode]

        df_soft = df[df["model"] == "Soft"].sort_values(by='K')
        if not df_soft.empty:
            plt.plot(df_soft['K'], df_soft['f1_macro'],
                     marker='o',
                     linestyle='--',
                     linewidth=2,
                     markersize=10,
                     color=base_color,
                     label=f"{mode.upper()} - Soft")

        df_non_soft = df[df["model"] != "Soft"]
        if not df_non_soft.empty:
            best_model_name = df_non_soft.loc[df_non_soft['f1_macro'].idxmax(), 'model']

            df_best_ns = df_non_soft[df_non_soft["model"] == best_model_name].sort_values(by='K')

            plt.plot(df_best_ns['K'], df_best_ns['f1_macro'],
                     marker='s',
                     linestyle='-',
                     linewidth=2,
                     markersize=10,
                     color=base_color,
                     label=f"{mode.upper()} - {best_model_name}")

    plt.xlabel("Dimensione del vocabolario", fontsize=12)
    plt.ylabel("F1 macro", fontsize=12)

    plt.legend(fontsize=11, loc="best", framealpha=0.9)

    plt.xticks([50, 100, 500])
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=18))
    plt.grid(axis='both', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    folders_to_analyze = ["gray_1_mln_features", "pca_1_mln_features"]
    compare_descriptors(folders_to_analyze)

    folders_to_analyze = ["pca_1_mln_features", "gray_1_mln_features"]
    compare_f1_scores(folders_to_analyze)
    compare_best_classifiers(folders_to_analyze)

    folders_to_compare = [
        "gray_100_k_features",
        "pca_100_k_features",
        "gray_1_mln_features",
        "gray_2_mln_features",
        "pca_1_mln_features",
        "pca_2_mln_features",
        # "pca_8_mln_features"
    ]
    compare_f1_with_variable_markers(folders_to_compare)
