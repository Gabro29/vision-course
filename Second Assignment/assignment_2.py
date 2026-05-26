
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize, StandardScaler, LabelEncoder, label_binarize
from sklearn.model_selection import StratifiedKFold

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.base import BaseEstimator, clone

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay



def compute_n_descriptors_upper_bound(total_descriptors: int=1_000_000) -> int:
    """
        Returns the number of feature descriptors for each image.
        Based on a number of totals descriptors for the whole dataset, like 100k, then the
        number of descriptors is calculated to spread them equally for each image.

        We assume each image will have for sure this number of feature descriptors, so the whole dataset will have
        the total number indicated in the argument of the function.
        In practise, is not possible to extract the same number of SIFT for all images.
        And also some images will have no SIFT.
    """

    class_image_counts_file = os.path.join(os.getcwd(), "checkpoints", "bovw", "image_for_each_class.pkl")
    if not os.path.exists(class_image_counts_file):
        AID_dir = os.path.join(os.getcwd(), "DATASET", "AID")
        class_image_counts = {}
        for class_name in os.listdir(AID_dir):
            class_path = os.path.join(AID_dir, class_name)
            image_count = sum(1 for f in os.listdir(class_path) if f.endswith(".jpg"))
            class_image_counts[class_name] = image_count

        pickle.dump(class_image_counts, open(class_image_counts_file, "wb"))

    class_image_counts = pickle.load(open(class_image_counts_file, "rb"))
    images_to_use = min(class_image_counts.values())
    # sift_per_image = total / (n_classi * numero_immagini_classe_minoritaria)
    descriptor_for_each_image = int(total_descriptors / (len(class_image_counts) * images_to_use))

    return descriptor_for_each_image


def get_min_descriptor_count() -> int:
    """
        After all feature descriptors are extracted from each class,
        we know exactly how many SIFT we have for each class.
        The class with less SIFT will force all the others to have the same number of SIFT.

        At the end, we return this minimum value of feature descriptors.
    """

    all_descriptors_dir = os.path.join(os.getcwd(), "checkpoints", "bovw", "descriptors")
    classes_with_descriptors_size = {}
    for filename in os.listdir(all_descriptors_dir):
        class_name = filename.split("_")[0]
        file_path = os.path.join(all_descriptors_dir, filename)
        data = np.load(file_path, mmap_mode='r')
        classes_with_descriptors_size[class_name] = data.shape[0]
    min_descriptors = min(classes_with_descriptors_size.values())

    return min_descriptors


def pre_process_img(current_image: np.ndarray, apply :str = "pca") -> np.ndarray:
    """
        Apply PCA (default) or Gray scale on a given image.

        For PCA:

            1. An image is a matrix nx3, where n is the number of pixels (HxW)
               and 3 is the number of channel (RGB)

            2. Extract the first principal component

            3. Project each pixel on the PC1 axis

            4. Normalize values in range 0-255
    """

    if apply == "pca":
        pixels = current_image.reshape(-1, 3).astype(np.float32)
        mean, eigenvectors = cv2.PCACompute(pixels, mean=None)
        max_idx = np.argmax(np.abs(eigenvectors[0]))
        if eigenvectors[0][max_idx] < 0:
            eigenvectors[0] = -eigenvectors[0]
        pc1 = np.dot(pixels - mean.flatten(), eigenvectors[0])
        pc1_img = pc1.reshape(current_image.shape[:2])
        processed_img = cv2.normalize(pc1_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    else:
        processed_img = cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)

    return processed_img


def collect_and_save_descriptors():
    """
        Save a file for each class.
        In each file there are all the feature descriptors extracted from each image of the class.

        Feature descriptors are also L2 normalized before being saved.

        Furthermore, also the name of images with no SIFT are saved into a file.
    """

    all_descriptors_dir = os.path.join(os.getcwd(), "checkpoints", "bovw", "descriptors")
    if not os.path.exists(all_descriptors_dir):
        os.makedirs(all_descriptors_dir)
        ft_ds_to_use = compute_n_descriptors_upper_bound()
        sift_obj = cv2.SIFT_create(ft_ds_to_use, contrastThreshold=0.04)

        AID_dir = os.path.join(os.getcwd(), "DATASET", "AID")
        images_with_no_sift = []
        for class_foldername, _, _ in os.walk(AID_dir):
            if class_foldername != AID_dir:
                X = []
                for foldername, _, filenames in os.walk(class_foldername):
                    for filename in filenames:
                        if filename.endswith("jpg"):
                            current_file = os.path.join(class_foldername, filename)
                            current_image = cv2.imread(current_file)
                            processed_img = pre_process_img(current_image)
                            _, current_sift_descriptors = sift_obj.detectAndCompute(processed_img, None)
                            if current_sift_descriptors is not None:
                                X.append(current_sift_descriptors)
                            else:
                                # Non vi sono SIFT per alcune immagini
                                images_with_no_sift.append(filename)

                X = np.vstack(X)
                # Normalize with L2 norm
                X = normalize(X, norm='l2', axis=1)
                # Save
                class_name = os.path.basename(class_foldername)
                path_tosave = os.path.join(all_descriptors_dir, f"{class_name}_descriptors.npy")
                np.save(path_tosave, X)

        if images_with_no_sift:
            images_with_no_SIFT_path = os.path.join(os.getcwd(), "checkpoints", "bovw", "images_with_no_SIFT.txt")
            with open(images_with_no_SIFT_path, 'a+') as f:
                for name in images_with_no_sift:
                    f.write(f"{name}\n")


def balance_and_combine_descriptors() -> int:
    """
        Group in all-in-one file all the descriptors of all the classes.
        Ensure also there are the same number of descriptors for each class, so they are balanced.

        It selects, among each class, the one with less samples thanks to the function get_min_descriptor_count().
        This will be the number of samples also for all the others.
        In this way we maintain our train set balanced among each class,
        without applying under or over sampling technique.

        Furthermore, if a class has more descriptors than the minimum found, we select randomly
        the descriptors to pick for that class. In this way there is no bias: we can pick descriptors
        from the first image with the same probability of the last.

        Returns how many feature descriptors has been picked for all the classes.
    """

    all_descriptors_path = os.path.join(os.getcwd(), "checkpoints", "bovw", "all_descriptors.npy")
    if not os.path.exists(all_descriptors_path):
        rng = np.random.default_rng(seed=177)
        all_descriptors_dir = os.path.join(os.getcwd(), "checkpoints", "bovw", "descriptors")

        num_ft_ds = get_min_descriptor_count()
        all_arrays = []
        for filename in os.listdir(all_descriptors_dir):
            file_path = os.path.join(all_descriptors_dir, filename)
            data = np.load(file_path)
            # Si fa una selezione random anche per la classe minoritaria,
            # in questo modo di introduce uno shuffle preventivo
            ft_indexes_sampled_to_use = rng.choice(np.arange(data.shape[0]), size=num_ft_ds, replace=False)
            all_arrays.append(data[ft_indexes_sampled_to_use])

        np.save(all_descriptors_path, np.concatenate(all_arrays, axis=0))

    X = np.load(all_descriptors_path, mmap_mode='r')
    return X.shape[0]


def compute_all_vocabularies():
    """
        Apply KMeans, with different values of K, to all the feature descriptors,
        then each model is saved.

        These will be our visual words.
    """

    Ks = [50, 100, 500]
    models_dir = os.path.join("checkpoints", "bovw", "cluster_models")
    descriptors_path = os.path.join("checkpoints", "bovw", "all_descriptors.npy")
    if not os.path.exists(models_dir):
        os.makedirs(models_dir, exist_ok=True)
        X = np.load(descriptors_path)
        for K in Ks:
            model_path = os.path.join(models_dir, f"cluster_model_{K}.pkl")
            cluster_model = KMeans(n_clusters=K, verbose=True, random_state=177)
            cluster_model.fit(X)
            pickle.dump(cluster_model, open(model_path, "wb"))


def get_bovw_histogram_l1_normalized(img_ft_ds: np.ndarray, cluster_model: KMeans, n_clusters: int) -> np.ndarray:
    """
        Simply calculate the L1 normalized histogram, so the bovw histogram for a given image.

        Note:
            during the vocabulary building phase, feature descriptors extracted from the AID dataset
            were L2-normalized before being fed into KMeans. For consistency, any new descriptor must
            undergo the same L2 normalization.
    """

    img_ft_ds = normalize(img_ft_ds, norm='l2', axis=1)
    bovw_hist = cluster_model.predict(img_ft_ds)

    # Con minlength=n_clusters forziamo l'array ad avere dimensione fissa,
    # riempiendo di zeri le occorrenze relative alle visual words che
    # difatto non sono state trovate in quella specifica immagine
    bovw_hist = np.bincount(bovw_hist, minlength=n_clusters)
    bovw_hist_norm = normalize(bovw_hist.reshape(1, -1), norm='l1', axis=1).flatten()

    return bovw_hist_norm


def create_dataset_with(n_clusters: int):
    """
        For the UCMerced_LandUse dataset.

        Given the number of cluster to use, it will load the respective K-Means model.
        Then it will calculate the bovw histogram L1 normalized vector for each image.

        At the end an entire dataset is built by pairing vectors and labels.

        Furthermore, also the name of images with no SIFT are saved into a file.
    """

    datasets_dir = os.path.join(os.getcwd(), "checkpoints", "classification", "datasets")
    path_tosave = os.path.join(datasets_dir, f"bovw_histograms_K_{n_clusters}.npz")

    if not os.path.exists(path_tosave):
        os.makedirs(datasets_dir, exist_ok=True)

        kmeans_path = os.path.join(os.getcwd(), "checkpoints", "bovw", "cluster_models", f"cluster_model_{n_clusters}.pkl")
        kmeans_model = pickle.load(open(kmeans_path, "rb"))

        UCMerced_dir = os.path.join(os.getcwd(), "DATASET", "UCMerced_LandUse", "Images")
        sift_obj = cv2.SIFT_create()

        bovw_histograms = []
        labels = []
        images_with_no_sift = []

        for class_foldername in os.listdir(UCMerced_dir):
            class_path = os.path.join(UCMerced_dir, class_foldername)

            for img_name in os.listdir(class_path):
                if img_name.endswith('.tif'):
                    current_file = os.path.join(class_path, img_name)
                    current_image = cv2.imread(current_file)
                    processed_img = pre_process_img(current_image)
                    _, current_sift_descriptors = sift_obj.detectAndCompute(processed_img, None)
                    if current_sift_descriptors is not None:
                        bovw_hist = get_bovw_histogram_l1_normalized(
                            img_ft_ds=current_sift_descriptors, cluster_model=kmeans_model, n_clusters=n_clusters
                        )

                        bovw_histograms.append(bovw_hist)
                        labels.append(class_foldername)
                    else:
                        images_with_no_sift.append(img_name)

        if images_with_no_sift:
            images_with_no_SIFT_path = os.path.join(datasets_dir, "images_with_no_SIFT.txt")
            with open(images_with_no_SIFT_path, 'a+') as f:
                for name in images_with_no_sift:
                    f.write(f"{name} ----- with n_clusters={n_clusters}\n")

        X = np.array(bovw_histograms)
        Y = np.array(labels)
        np.savez(path_tosave, X=X, Y=Y)


def classifiers_to_use() -> dict:
    """
        Define all models to use and their hyperparameters.
    """

    clf1 = SVC(kernel="linear", probability=True, class_weight="balanced", random_state=177)
    clf2 = RandomForestClassifier(n_estimators=500, class_weight="balanced", random_state=177)
    clf3 = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=177)
    clf4 = SVC(kernel="rbf", probability=True, C=10.0, class_weight="balanced", random_state=177)
    clf5 = KNeighborsClassifier(weights="distance", metric="cosine")
    clf6 = VotingClassifier(voting='soft', estimators=[('svm_lin', clone(clf1)), ('svm_rbf', clone(clf4)), ('rf', clone(clf2))])

    classifiers_dict = {
        "SVM_Linear": clf1,
        "RandomForest": clf2,
        "LogReg": clf3,
        "SVM_RBF": clf4,
        "KNN": clf5,
        "Soft": clf6
    }

    return classifiers_dict


def get_model_by_name(model_name: str) -> BaseEstimator:
    """
        Given model name, returns the relative object.
    """

    classifiers_dict = classifiers_to_use()

    return classifiers_dict[model_name]


def train_and_get_model_performance(X: np.ndarray, y: np.ndarray, ref_model: BaseEstimator, n_classes: int) -> tuple:
    """
        Using a 3-fold cross-validation with StratifiedKFold,
        we make sure the folds are made by preserving the percentage of
        samples for each class.

        For each fold, we save in a list the prediction made. So, at the end, we will have the prediction
        for all the train set. On that prediction we calculate f1_macro and auc_macro one versus rest.

        It returns a dictionary with the metrics and the confusion matrix.
    """


    skfolds = StratifiedKFold(n_splits=3, shuffle=True, random_state=177)

    y_true_ordered_by_fold = []
    y_pred_ordered_by_fold = []
    y_proba_ordered_by_fold = []

    for train_index, test_index in skfolds.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        scaler = StandardScaler()
        X_train_std = scaler.fit_transform(X_train)
        X_test_std = scaler.transform(X_test)

        model = clone(ref_model)
        model.fit(X_train_std, y_train)

        y_pred = model.predict(X_test_std)
        y_proba = model.predict_proba(X_test_std)

        y_true_ordered_by_fold.extend(y_test)
        y_pred_ordered_by_fold.extend(y_pred)
        y_proba_ordered_by_fold.extend(y_proba)

    y_true_ordered_by_fold = np.array(y_true_ordered_by_fold)
    y_pred_ordered_by_fold = np.array(y_pred_ordered_by_fold)
    y_proba_ordered_by_fold = np.array(y_proba_ordered_by_fold)

    metrics = {"f1_macro": f1_score(y_true_ordered_by_fold, y_pred_ordered_by_fold, average="macro")}

    y_true_bin = label_binarize(y_true_ordered_by_fold, classes=range(n_classes))
    metrics["auc_ovr"] = roc_auc_score(y_true_bin, y_proba_ordered_by_fold, multi_class='ovr', average='macro')

    cm = confusion_matrix(y_true_ordered_by_fold, y_pred_ordered_by_fold)

    return metrics, cm


def evaluate_models():
    """
        Perform the training with all the models, then produce a summary file.
    """

    path_tosave = os.path.join(os.getcwd(), "checkpoints", "classification", "model_summary.csv")
    if os.path.exists(path_tosave):
        return

    models = classifiers_to_use()

    Ks = [50, 100, 500]
    results = []

    for K in Ks:
        path = os.path.join(os.getcwd(), "checkpoints", "classification", "datasets", f"bovw_histograms_K_{K}.npz")
        data = np.load(path)
        X, y = data['X'], data['Y']

        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        target_names = le.classes_
        n_classes = len(le.classes_)

        for name, base_model in models.items():

            metrics, cm = train_and_get_model_performance(X, y_encoded, base_model, n_classes)

            results.append({
                "K": K,
                "model": name,
                "f1_macro": metrics["f1_macro"],
                "auc_ovr": metrics["auc_ovr"]
            })

            fig, ax = plt.subplots(figsize=(8, 6))
            ConfusionMatrixDisplay(cm, display_labels=target_names).plot(ax=ax, xticks_rotation='vertical', cmap="hot")
            plt.title(f"CM: {name} (K={K})")
            cm_path = os.path.join(os.getcwd(), "checkpoints", "classification", "CM")
            os.makedirs(cm_path, exist_ok=True)
            plt.savefig(os.path.join(cm_path, f"confusion_matrix_{name}_K{K}.png"), dpi=300, bbox_inches='tight')
            # plt.show()

    results = pd.DataFrame(results)
    results.sort_values(by=['f1_macro', 'auc_ovr'], ascending=[False, False], inplace=True)
    results.to_csv(path_tosave, index=False)


def save_best_model():
    """
        Put into one object all stuffs needed for classification: encoder, scaler, model, n_clusters
    """

    best_configuration_path = os.path.join(os.getcwd(), "best_configuration.pkl")
    if os.path.exists(best_configuration_path):
        return

    # Get best configuration
    summary_path = os.path.join(os.getcwd(), "checkpoints", "classification", "model_summary.csv")
    summary = pd.read_csv(summary_path, engine="c")
    summary.sort_values(by=['f1_macro', 'auc_ovr'], ascending=[False, False], inplace=True)
    best_configuration = summary.iloc[0]

    # Train again on the whole dataset
    best_model = get_model_by_name(best_configuration["model"])
    path = os.path.join(os.getcwd(), "checkpoints", "classification", "datasets", f"bovw_histograms_K_{best_configuration['K']}.npz")
    data = np.load(path)
    X, y = data['X'], data['Y']
    scaler = StandardScaler().fit(X)
    le = LabelEncoder().fit(y)
    X_scaled = scaler.transform(X)
    y_encoded = le.transform(y)
    best_model.fit(X_scaled, y_encoded)

    # Get visual words
    kmeans_path = os.path.join(os.getcwd(), "checkpoints", "bovw", "cluster_models", f"cluster_model_{best_configuration['K']}.pkl")
    kmeans_model = pickle.load(open(kmeans_path, "rb"))

    # Save into a single object
    model_bundle = {
        "visual_words": kmeans_model,
        "K": best_configuration["K"],
        "model": best_model,
        "scaler": scaler,
        "label_encoder": le
    }

    pickle.dump(model_bundle, open(best_configuration_path, "wb"))


def single_img_classification(img_path: str) -> str:
    """
        Perform the classification task on a given image using the best model obtained.
    """

    best_configuration_path = os.path.join(os.getcwd(), "best_configuration.pkl")
    model_bundle = pickle.load(open(best_configuration_path, "rb"))

    current_image = cv2.imread(img_path)

    # Build descriptors
    sift_obj = cv2.SIFT_create()
    processed_img = pre_process_img(current_image)
    _, current_sift_descriptors = sift_obj.detectAndCompute(processed_img, None)
    if current_sift_descriptors is None:
        return "Nessuna SIFT trovata per l'immagine inserita - impossibile effettuare la classificazione"

    # Get BOVW Histogram
    kmeans_model = model_bundle["visual_words"]
    n_clusters = model_bundle["K"]
    bovw_hist = get_bovw_histogram_l1_normalized(
        img_ft_ds=current_sift_descriptors, cluster_model=kmeans_model, n_clusters=n_clusters
    )

    # Make prediction
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    le = model_bundle["label_encoder"]

    bovw_hist_std = scaler.transform(bovw_hist.reshape(1, -1))
    y_pred = model.predict(bovw_hist_std)
    label = le.inverse_transform(y_pred)[0]

    return label


if __name__ == "__main__":

    training = False

    if training:
        # Part 1 - Build vocabulary
        collect_and_save_descriptors()
        n_ft_ds = balance_and_combine_descriptors()
        print(f"Numero di feature descriptors utilizzate per costruire il vocabolario con KMeans: {n_ft_ds}")
        compute_all_vocabularies()

        # Part 2 - Build dataset for classification
        for K in (50, 100, 500):
            create_dataset_with(n_clusters=K)

        # Part 3 - Classification and performance
        evaluate_models()

        # Part 4 - Model selection
        save_best_model()

    # Part 5 - Inference
    img_path = r"punta_raisi.png"
    print(f"\n<- Classifico {os.path.basename(img_path)} ->")
    label = single_img_classification(img_path)
    print(f"<- L'immagine fornita appartiene alla classe: {label} ->")
