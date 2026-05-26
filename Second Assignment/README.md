
# User Guide

The idea is to place the `assignment_2.py` script in a folder along with the `best_configuration.pkl` file.
The `DATASET` folder is optional. IMPORTANT: unpack the `best_configuration.zip` to obtain `best_configuration.pkl` file.
```
main_dir:
|
|--DATASET (folder)
|    |
|    |--AID (folder)
|    |--UC Merced (folder
|
|--assignment_2.py
|--best_configuration.pkl
|--plots_assignment_2.py
```

- The `assignment_2.py` script allows you to set a flag if you want to start training from scratch. 
Setting `training = False` will skip directly to the inference phase.


- The `assignment_2.py` script contains a series of modular functions that can be used independently. You can read the docstring to understand what each function does or use the help method --> help(method).


- The `plots_assignment_2.py` script was used to create the graphs included in the presentation.

---

## Usage Example

### Single Use

Simply enter the path to an image. You can also specify a preprocessing strategy. 
If nothing is specified, the strategy is assumed to be *pca*. 
As a preprocessing strategy, you can choose one of the following:
- gray 
- pca


Here's how to specify the preprocessing strategy: go to the `pre_process_img` function and set the `apply` parameter to `“gray”`.
```
def pre_process_img(current_image: np.ndarray, apply :str = “gray”) -> np.ndarray
```

Then you can call the `single_img_classification` method.

```
img_path = r“punta_raisi.png”
print(f“\n<- Classifying {os.path.basename(img_path)} ->”)
label = single_img_classification(img_path)
print(f“<- The provided image belongs to the class: {label} ->”)
```


### Complete Usage

Below are the procedures performed that led to the stated results.
```
training = True

if training:
    # Part 1 - Build vocabulary
    collect_and_save_descriptors()
    n_ft_ds = balance_and_combine_descriptors()
    print(f“Number of feature descriptors used to build the vocabulary with K-Means: {n_ft_ds}”)
    compute_all_vocabularies()

    # Part 2 - Build dataset for classification
    for K in (50, 100, 500):
        create_dataset_with(n_clusters=K)

    # Part 3 - Classification and performance
    evaluate_models()

    # Part 4 - Model selection
    save_best_model()

# Part 5 - Inference
img_path = r“punta_raisi.png”
print(f“\n<- Classifying {os.path.basename(img_path)} ->”)
label = single_img_classification(img_path)
print(f“<- The provided image belongs to the class: {label} ->”)
```


---
#### Required libraries:
Python version _3.14.2_.
- pandas
- numpy
- matplotlib
- sklearn
- opencv-python
---

#### Notes

The `best_configuration.pkl` file is a bundle of all the objects needed for classification. 
Specifically, it is a Python dictionary that stores:
- The weights of the best classifier
- The number of visual words and the weights of the K-means model
- The encoder used to process the categorical labels to be classified
- The StandardScaler trained on the entire training dataset


### Reproducibility
To ensure full reproducibility of the experiments, a random seed with value 177 was set.
Specifically, this seed is used in the following functions:
- `balance_and_combine_descriptors`: used in `np.random.default_rng(seed=177)`. This ensures that the exact subset of data is produced on every run. 
- `compute_all_vocabularies`: passed as the parameter `random_state=177` to the KMeans model.
- `classifiers_to_use`: passed as the parameter `random_state=177` during classifier initialization.
- `train_and_get_model_performance`: passed as the parameter `random_state=177` to StratifiedKFold to control the shuffle.
