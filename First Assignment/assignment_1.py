
import cv2
import numpy as np
import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
from math import fsum
import os


def compute_mi(img_ref: np.ndarray, img_mov: np.ndarray, bins: int) -> float:
    """The reference formula is MI(Ir, Im) = H(Ir) + H(Im) − H(Ir, Im)"""

    # Con che frequenza assoluta la coppia (intensità_pixel_ref, intensità_pixel_mov) si è manifestata
    # 256 poiché il range fa end-1
    h_joint, _, _ = np.histogram2d(img_ref.ravel(), img_mov.ravel(), bins=bins, range=[[0, 256], [0, 256]])

    # Frequenza relativa ossia la probabilità
    p_joint = h_joint / np.sum(h_joint)

    # Marginalizzo
    p_ref = np.sum(p_joint, axis=1)
    p_mov = np.sum(p_joint, axis=0)

    p_ref_nz = p_ref[p_ref > 0]
    p_mov_nz = p_mov[p_mov > 0]
    p_joint_nz = p_joint[p_joint > 0]

    # Calcolo delle entropie
    h_ref = -np.sum(p_ref_nz * np.log2(p_ref_nz))
    h_mov = -np.sum(p_mov_nz * np.log2(p_mov_nz))
    h_joint_entropy = -np.sum(p_joint_nz * np.log2(p_joint_nz))

    mi = h_ref + h_mov - h_joint_entropy

    return mi


def neg_mi(params: list, img_ref: np.ndarray, img_mov: np.ndarray, bins: int) -> float:
    """Custom function to minimize, used by optimizer"""

    tx, ty, theta = params
    h, w = img_ref.shape
    cx, cy = w / 2.0, h / 2.0
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Ruotando intorno al centro dell'immagine
    T = np.array([
        [cos_t, -sin_t, (1 - cos_t) * cx + sin_t * cy + tx],
        [sin_t, cos_t, -sin_t * cx + (1 - cos_t) * cy + ty]
    ], dtype=np.float32)

    aligned = cv2.warpAffine(img_mov, T, (w, h), flags=cv2.INTER_LINEAR)

    # Maschera per escludere i pixel artificialmente neri dovuti alla trasformazione nel calcolo della MI
    mask = cv2.warpAffine(np.full((h, w), 255, dtype=np.uint8), T, (w, h), flags=cv2.INTER_NEAREST)
    valid = mask > 0

    # L'immagine trasformata è tutta nera, quindi la trasformazione è stata troppo estrema
    if len(img_ref[valid]) == 0:
        return 1e8

    mi_value = compute_mi(img_ref[valid], aligned[valid], bins)

    return -mi_value


def convert_center_to_origin(tx_center: float, ty_center: float, theta: float, shape: tuple) -> tuple:
    """Converts translation parameters, calculated with respect to the center of the image, into the origin system of the image (left-top origin)"""
    h, w = shape
    cx = w / 2.0
    cy = h / 2.0

    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    tx_origin = tx_center + cx - cx * cos_t + cy * sin_t
    ty_origin = ty_center + cy - cx * sin_t - cy * cos_t

    return tx_origin, ty_origin


def pipeline(reference_img: np.ndarray, moving_img: np.ndarray, n_bin: int, method: str, x0: np.ndarray, decimal_digits: int) -> tuple:
    """
    Takes two images as matrix nxn and returns the best parameters
    that performs the best transformation T, according to the given method and the number of bins.
    The best T is obtained by maximizing the Mutual Information of the two images.
    """

    method_lower = method.lower()
    options = {'disp': False}

    if method_lower == "nelder-mead":
        initial_simplex = np.array([
            [x0[0], x0[1], x0[2]],
            [x0[0] + 5.0, x0[1], x0[2]],
            [x0[0], x0[1] + 5.0, x0[2]],
            [x0[0], x0[1], x0[2] + np.deg2rad(5)],
        ])
        options |= {'initial_simplex': initial_simplex, 'adaptive': False}

    # elif method_lower == "bfgs":
    #     step_theta = np.deg2rad(0.5)
    #     eps_steps = np.array([0.5, 0.5, step_theta])
    #     options |= {'eps': eps_steps, 'gtol': 1e-4}

    res = optimize.minimize(
        neg_mi,
        x0,
        args=(reference_img, moving_img, n_bin),
        method=method,
        options=options
    )

    tx_pred = round(float(res.x[0]), decimal_digits)
    ty_pred = round(float(res.x[1]), decimal_digits)
    theta_pred = round(float(res.x[2]), decimal_digits)

    return tx_pred, ty_pred, theta_pred


def pre_processing_img(set_type: str, dir_name: str, img_name: str, tipo: str, prep_strategy: str) -> np.ndarray:
    """
        Takes an image and apply a preprocess strategy:
            - GRAYSCALE
            - HSV -> extract only the V channel
            - PC1 of PCA -> Projects along the direction of the first principal component

        Optionally apply a gaussian filter with a custom kernel.
    """

    path = os.path.join(os.getcwd(), "DATASET", set_type, dir_name)
    img_path = os.path.join(path, img_name + f"_{tipo}.png")
    img = cv2.imread(img_path)

    base = prep_strategy.split("-")[0]

    if base in ("gray", "none"):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    elif base == "hsv_v":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        img = img[:, :, 2]

    elif base == "pca":
        pixels = img.reshape(-1, 3).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(pixels, mean=None)
        max_idx = np.argmax(np.abs(eigenvectors[0]))
        if eigenvectors[0][max_idx] < 0:
            eigenvectors[0] = -eigenvectors[0]
        pc1 = np.dot(pixels - mean.flatten(), eigenvectors[0])
        pc1_img = pc1.reshape(img.shape[:2])
        img = cv2.normalize(pc1_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    if "gauss" in prep_strategy:
        n = int(prep_strategy.split("-")[-1])
        img = cv2.GaussianBlur(img, (n, n), 0)

    return img


def gridsearch_on_validation(dir_name: str, img_name: str, true_params: tuple) -> pd.DataFrame:
    """Process a couple of images and returns T's parameters for each configuration."""

    # Start point: [tx, ty, theta]
    x0 = np.array([0.0, 0.0, 0.0])
    decimal_digits = 5
    preprocessing_strategies = ["gray", "hsv_v", "pca",
                                "gray-gauss-3", "hsv_v-gauss-3", "pca-gauss-3",
                                "gray-gauss-5", "hsv_v-gauss-5", "pca-gauss-5",
                                "gray-gauss-7", "hsv_v-gauss-7", "pca-gauss-7"]
    bins = [64, 128, 256]
    methods = ["powell", "Nelder-Mead"] # , "BFGS"]
    combinations_for_current_couple = []

    # GridSearch
    for prep_strategy in preprocessing_strategies:
        reference_img = pre_processing_img("val", dir_name, img_name, "R", prep_strategy)
        moving_img = pre_processing_img("val", dir_name, img_name, "T", prep_strategy)

        for n_bin in bins:
            for method in methods:

                tx_pred, ty_pred, theta_pred = pipeline(reference_img, moving_img, n_bin, method, x0, decimal_digits)

                tx_pred, ty_pred = convert_center_to_origin(tx_pred, ty_pred, theta_pred, reference_img.shape)

                combinations_for_current_couple.append({
                                "image_couple_name": img_name,
                                "pair": dir_name,
                                "preprocessing": prep_strategy,
                                "bin": n_bin,
                                "method": method,

                                "tx": tx_pred,
                                "residuo_tx": round(fsum([tx_pred, -true_params[0]]), decimal_digits),

                                "ty": ty_pred,
                                "residuo_ty": round(fsum([ty_pred, -true_params[1]]), decimal_digits),

                                "theta": theta_pred,
                                "residuo_theta": round(((theta_pred - true_params[2] + np.pi) % (2 * np.pi)) - np.pi, decimal_digits),

                                "residuo_diag": np.sqrt((tx_pred - true_params[0])**2 + (ty_pred - true_params[1])**2)
                })

    return pd.DataFrame(combinations_for_current_couple)


def analyze_val_set(output_file: str) -> pd.DataFrame:
    """
    Process each couple of images in the val directory and returns a Dataframe with the results of all configurations.
    The Dataframe is also saved in a csv file.
    """

    all_configurations = []

    GT = pd.read_csv("DATASET/GT.csv", sep=";", engine="c")

    for foldername, _, filenames in os.walk("DATASET"):
        if "val" in foldername:
            for filename in filenames:
                if filename.endswith("_R.png"):

                    dir_name = os.path.basename(foldername)
                    image_name = filename.replace("_R.png", "")

                    current_image_row = GT[(GT["Filename"] == image_name) & (GT["Pair"] == dir_name)]

                    true_params = current_image_row["Tx"].iloc[0], current_image_row["Ty"].iloc[0], current_image_row["AngleRad"].iloc[0]

                    current_couple_configuration = gridsearch_on_validation(dir_name, image_name, true_params)

                    all_configurations.append(current_couple_configuration)


    result = pd.concat(all_configurations, ignore_index=True)
    result.to_csv(f"{output_file}.csv", index=False)
    return result


def get_best_params(df: pd.DataFrame) -> tuple:
    """
    Analyze all_configurations and compute metrics.
    Since there are parameters with different measures unit, two type of sorting are done.
    The best method is chosen looking for the one that appears in the top of both sorting.
    """

    df = df.copy()
    df['abs_tx'] = df['residuo_tx'].abs()
    df['abs_ty'] = df['residuo_ty'].abs()
    df['abs_theta'] = df['residuo_theta'].abs()

    summary = df.groupby(['preprocessing', 'method', 'bin']).agg(
        mae_tx=('abs_tx', 'mean'),
        std_tx=('abs_tx', 'std'),

        mae_ty=('abs_ty', 'mean'),
        std_ty=('abs_ty', 'std'),

        mae_theta=('abs_theta', 'mean'),
        std_theta=('abs_theta', 'std'),

        mae_diag=("residuo_diag", "mean"),
        std_diag=("residuo_diag", "std")
    ).reset_index()
    summary = summary.round(5)

    df_trasl = summary.sort_values(by=['mae_diag', "mae_theta"]).reset_index(drop=True)
    df_trasl.to_csv("sorted_by_trasl.csv", index=False)
    df_theta = summary.sort_values(by=['mae_theta', "mae_diag"]).reset_index(drop=True)
    df_theta.to_csv("sorted_by_theta.csv", index=False)

    # Get automatically the best configuration, instead of looking directly into the csv files
    key_cols = ['preprocessing', 'method', 'bin']
    n = len(summary)
    top_k = 1
    while top_k <= n:
        top_diag_ids = set(df_trasl.head(top_k)[key_cols].apply(tuple, axis=1))
        for _, row in df_theta.head(top_k).iterrows():
            if tuple(row[key_cols]) in top_diag_ids:
                return row['preprocessing'], row['method'], int(row['bin'])
        top_k += 1
    raise ValueError("Nessuna configurazione ottimale trovata")


def plot_residui(df: pd.DataFrame):
    """
    Show the residuals plot. Also, a reference is drawn at y=0.
    """
    residual_cols = ['residuo_tx', 'residuo_ty', 'residuo_theta']
    titles = ['Grafico Residui Traslazione lungo X', 'Grafico Residui Traslazione lungo Y', 'Grafico Residui Rotazione']
    colors = ['#2A75D3', '#10A37F', '#EF476F']

    for i, col in enumerate(residual_cols):
        plt.figure(figsize=(10, 4))
        plt.scatter(range(len(df)), df[col], color=colors[i], s=60, edgecolor='black', linewidth=0.8, alpha=0.85, label='Residuo')
        plt.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.8)

        plt.title(titles[i], fontweight='bold', fontsize=12, pad=10)

        if 'theta' in col:
            plt.ylabel('Residuo (rad)')
        else:
            plt.ylabel('Residuo (pixel)')

        plt.xlabel('Coppia Immagini')
        plt.grid(True, linestyle=':', alpha=0.7, color='gray')
        plt.xticks(range(len(df)), df['image_couple_name'], rotation=45, ha='right')
        y_min, y_max = df[col].min(), df[col].max()
        plt.yticks(np.linspace(y_min, y_max, 15))
        plt.tight_layout()
        plt.show()


def visualize_results(df: pd.DataFrame, set_type: str):
    """
    Shows both aligned and difference image, press a button to skip to the next couple of images.
    """

    for _, row in df.iterrows():
        dir_name = row['pair']
        img_name = row['image_couple_name']
        tx = row['tx']
        ty = row['ty']
        theta = row['theta']
        prep_strategy = row['preprocessing']

        img_ref = pre_processing_img(set_type, dir_name, img_name, "R", prep_strategy)
        img_mov = pre_processing_img(set_type, dir_name, img_name, "T", prep_strategy)
        h, w = img_ref.shape

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        T = np.array([
            [cos_t, -sin_t, tx],
            [sin_t, cos_t, ty]
        ], dtype=np.float32)

        aligned = cv2.warpAffine(img_mov, T, (w, h), flags=cv2.INTER_LINEAR)

        diff = cv2.absdiff(img_ref, aligned)

        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        fig.canvas.manager.set_window_title(f"Pair: {dir_name} | Prep: {prep_strategy} | Method: {row['method']} (Bins: {row['bin']})")

        # Reference
        # axes[0].imshow(img_ref, cmap='gray', vmin=0, vmax=255)
        # axes[0].set_title("1. Reference (Ir)")
        # axes[0].axis('off')

        # Moving
        # axes[1].imshow(img_mov, cmap='gray', vmin=0, vmax=255)
        # axes[1].set_title("2. Moving Original (Im)")
        # axes[1].axis('off')

        # Aligned
        axes[0].imshow(aligned, cmap='gray', vmin=0, vmax=255)
        axes[0].set_title(f"Aligned")
        axes[0].axis('off')

        # Differenza
        axes[1].imshow(diff, cmap='gray', vmin=0, vmax=255)
        axes[1].set_title("Ref - Aligned")
        axes[1].axis('off')

        plt.tight_layout()
        plt.show(block=False)
        plt.waitforbuttonpress()
        plt.close()


def plot_andamento_mi(dir_name: str, img_name: str, bins: int, method: str, set_type: str, prep_strategy: str):
    """
    Based on given hyperparameters plots MI trend by iteration for a given image.
    """

    def mi_callback(xk):
        """Called by optimizer to save the MI during iterations."""
        current_mi = -neg_mi(xk, reference_img, moving_img, bins)
        mi_history.append(current_mi)


    reference_img = pre_processing_img(set_type, dir_name, img_name, "R", prep_strategy)
    moving_img = pre_processing_img(set_type, dir_name, img_name, "T", prep_strategy)

    x0 = np.array([0.0, 0.0, 0.0])
    method_lower = method.lower()
    options = {'disp': False}

    if method_lower == "nelder-mead":
        initial_simplex = np.array([
            [x0[0], x0[1], x0[2]],
            [x0[0] + 5.0, x0[1], x0[2]],
            [x0[0], x0[1] + 5.0, x0[2]],
            [x0[0], x0[1], x0[2] + np.deg2rad(5)],
        ])
        options |= {'initial_simplex': initial_simplex, 'adaptive': False}

    mi_history = []

    optimize.minimize(
        neg_mi,
        x0,
        args=(reference_img, moving_img, bins),
        method=method,
        callback=mi_callback,
        options=options
    )

    if not mi_history:
        print("Nessuna iterazione registrata.")
        return

    plt.figure(figsize=(9, 5))
    plt.plot(range(1, len(mi_history) + 1), mi_history, marker='o', markersize=6, linestyle='-', linewidth=2,
             color='#1f77b4', markerfacecolor='white', markeredgewidth=1.5, label='Valore MI')

    plt.xlabel('Iterazione', fontsize=10)
    plt.ylabel('Mutua Informazione', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    mi_min, mi_max = min(mi_history), max(mi_history)
    plt.yticks(np.linspace(mi_min, mi_max, 10))
    plt.tight_layout()
    plt.show()


def analyze_test_set(output_file: str, best_prep_strategy: str, best_method: str, best_n_bin: int) -> pd.DataFrame:
    """
    Push each couple of images of the test set into the pipeline and returns a Dataframe with the results of all configurations.
    The Dataframe is also saved in a csv file.
    """

    all_configurations = []
    x0 = np.array([0.0, 0.0, 0.0])
    decimal_digits = 5

    GT = pd.read_csv("DATASET/GT.csv", sep=";", engine="c")

    for foldername, _, filenames in os.walk("DATASET"):
        if "test" in foldername:
            for filename in filenames:
                if filename.endswith("_R.png"):

                    dir_name = os.path.basename(foldername)
                    image_name = filename.replace("_R.png", "")

                    current_image_row = GT[(GT["Filename"] == image_name) & (GT["Pair"] == dir_name)]

                    true_params = current_image_row["Tx"].iloc[0], current_image_row["Ty"].iloc[0], current_image_row["AngleRad"].iloc[0]

                    reference_img = pre_processing_img("test", dir_name, image_name, "R", best_prep_strategy)
                    moving_img = pre_processing_img("test", dir_name, image_name, "T", best_prep_strategy)

                    tx_pred, ty_pred, theta_pred = pipeline(reference_img, moving_img, best_n_bin, best_method, x0, decimal_digits)

                    tx_pred, ty_pred = convert_center_to_origin(tx_pred, ty_pred, theta_pred, reference_img.shape)

                    all_configurations.append({
                        "image_couple_name": image_name,
                        "pair": dir_name,
                        "preprocessing": best_prep_strategy,
                        "bin": best_n_bin,
                        "method": best_method,

                        "tx": tx_pred,
                        "residuo_tx": round(fsum([tx_pred, -true_params[0]]), decimal_digits),

                        "ty": ty_pred,
                        "residuo_ty": round(fsum([ty_pred, -true_params[1]]), decimal_digits),

                        "theta": theta_pred,
                        "residuo_theta": round(((theta_pred - true_params[2] + np.pi) % (2 * np.pi)) - np.pi,
                                               decimal_digits),

                        "residuo_diag": np.sqrt((tx_pred - true_params[0])**2 + (ty_pred - true_params[1])**2)
                    })


    result = pd.DataFrame(all_configurations)
    result.to_csv(f"{output_file}.csv", index=False)

    return result


def calculate_average_on_test(output_file: str, df: pd.DataFrame, best_prep_strategy: str, best_method: str, best_n_bin: int, decimal_digits: int = 5):
    """
        Results on test set are used to compute mean results.
        The summary obtained is saved in a csv file.
    """

    df['abs_tx'] = df['residuo_tx'].abs()
    df['abs_ty'] = df['residuo_ty'].abs()
    df['abs_theta'] = df['residuo_theta'].abs()

    summary = pd.DataFrame([{
        'preprocessing': best_prep_strategy,
        'method': best_method,
        'bin': best_n_bin,

        'mae_tx': df['abs_tx'].mean(),
        'std_tx': df['abs_tx'].std(),

        'mae_ty': df['abs_ty'].mean(),
        'std_ty': df['abs_ty'].std(),

        'mae_theta': df['abs_theta'].mean(),
        'std_theta': df['abs_theta'].std(),

        'mae_diag': df['residuo_diag'].mean(),
        'std_diag': df['residuo_diag'].std(),
    }])

    summary = summary.round(decimal_digits)
    summary.to_csv(f"{output_file}_average.csv", index=False)


def single_use_pipeline(img_path: str, n_bin: int, method: str, prep_strategy: str="none") -> pd.DataFrame:

    x0 = np.array([0.0, 0.0, 0.0])

    filename = os.path.basename(img_path)
    image_name = filename.replace("_R.png", "")
    dir_name = os.path.basename(os.path.dirname(img_path))
    set_type = os.path.basename(os.path.dirname(os.path.dirname(img_path)))

    reference_img = pre_processing_img(set_type, dir_name, image_name, "R", prep_strategy)
    moving_img = pre_processing_img(set_type, dir_name, image_name, "T", prep_strategy)

    tx_pred, ty_pred, theta_pred = pipeline(reference_img, moving_img, n_bin, method, x0, decimal_digits=5)
    tx_pred, ty_pred = convert_center_to_origin(tx_pred, ty_pred, theta_pred, reference_img.shape)
    single_configuration = [{
        "image_couple_name": image_name,
        "pair": dir_name,
        "preprocessing": prep_strategy,
        "bin": n_bin,
        "method": method,
        "tx": tx_pred,
        "ty": ty_pred,
        "theta": theta_pred
    }]

    result = pd.DataFrame(single_configuration)

    visualize_results(result, set_type)

    return result


if __name__ == "__main__":

    # Singolo utilizzo
    single_use_pipeline(img_path=r"S:\Coding\Python\visione\Assignment_1\DATASET\test\c4\zh3_03_02_R.png",
                        n_bin=64, method="Nelder-Mead", prep_strategy="gray")

    # Fase di tuning degli iperparametri
    output_file = "all_configurations"
    df = analyze_val_set(output_file)

    # Metriche e Selezione
    # df = pd.read_csv("all_configurations.csv")
    best_prep, best_method, best_bin = get_best_params(df)
    best_config_df = df[(df['preprocessing'] == best_prep) & (df['method'] == best_method) & (df['bin'] == best_bin)]

    # Residui metodo migliore e visualizzazione
    plot_residui(best_config_df)
    visualize_results(best_config_df, "val")


    # Grafici per slide
    # config_df = df[(df['preprocessing'] == "gray-gauss-5") & (df['method'] == "Nelder-Mead") & (df['bin'] == 128)]
    # visualize_results(config_df, "val")
    # config_df = df[(df['preprocessing'] == "hsv_v") & (df['method'] == "powell") & (df['bin'] == 256)]
    # visualize_results(config_df, "val")


    # Andamento MI
    plot_andamento_mi("c3", "zh3_02_02", best_bin, best_method, "val", best_prep)
    plot_andamento_mi("c6", "zh1_01_02", best_bin, best_method, "val", best_prep)


    # Valutazione sul test set
    output_file = "test_eval"
    df = analyze_test_set(output_file, best_prep, best_method, best_bin)
    calculate_average_on_test(output_file, df, best_prep, best_method, best_bin)
    plot_residui(df)
    visualize_results(df, "test")
