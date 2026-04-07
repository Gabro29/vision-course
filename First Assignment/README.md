
# User Guide

The idea is to place this script in a folder and also put the `DATASET` folder in that same folder.

```
main_dir:
|
|--DATASET (folder)
|    |
|    |--val (folder)
|    |--test (folder
|    |--GT.csv
|
|--assignment_1.py
|
```

The script is designed to automatically analyze the structure of the `DATASET` folder, assuming it contains two subfolders: `test` and `val`. The `GT.csv` file is also assumed to be located within the `DATASET` folder.

The script contains a series of modular functions that can be used independently. You can read the doc string to understand what each function does, or use the help method --> help(method).

---

## Example of Use

### Single Use

You can enter the path to an image, whether it is a reference or a moving image. In fact, only the image name will be extracted, and the suffix `_R.png` or `_T.png` will be automatically appended as appropriate. You can also choose one of the available methods: *Powell* or *Nelder-Mead*.

You can also specify a preprocessing strategy. If nothing is specified, the strategy is assumed to be *gray*, meaning the image is converted to grayscale.
```
single_use_pipeline(img_path="path\of\your\image", n_bin=some_int, method="Nelder-Mead or Powell")
```

As a preprocessing strategy, you can choose one of the following:
- gray 
- hsv_v
- pca
- gray-gauss-3
- hsv_v-gauss-3
- pca-gauss-3
- gray-gauss-5
- hsv_v-gauss-5
- pca-gauss-5
- gray-gauss-7
- hsv_v-gauss-7
- pca-gauss-7

Here's how to specify the preprocessing strategy
```
single_use_pipeline(img_path="path\of\your\image", n_bin=some_int, method="Nelder-Mead or Powell", prep_strategy="hsv_v-gauss-5")
```

### Full Description

The following procedures were performed to achieve the stated results.
```
# Hyperparameter tuning phase
output_file = "all_configurations"
df = analyze_val_set(output_file)

# Metrics and Selection
df = pd.read_csv("all_configurations.csv")
best_prep, best_method, best_bin = get_best_params(df)
best_config_df = df[(df['preprocessing'] == best_prep) & (df['method'] == best_method) & (df['bin'] == best_bin)]

# Residues: Best Method and Visualization
plot_residui(best_config_df)
visualize_results(best_config_df, "val")


# MI Trend
plot_andamento_mi("c3", "zh3_03_01", best_bin, best_method, "test", best_prep)
plot_andamento_mi("c4", "zh3_03_02", best_bin, best_method, "test", best_prep)


# Test set evaluation
output_file = "test_eval"
df = analyze_test_set(output_file, best_prep, best_method, best_bin)
calculate_average_on_test(output_file, df, best_prep, best_method, best_bin)
plot_residui(df)
visualize_results(df, "test")
```


---
#### Required libraries:
Version of Python _3.14.2_.
- pandas
- numpy
- matplotlib
- scipy
- opencv-python

---

#### Notes

The `all_configurations.csv` file contains the results of the entire grid search; it has been included in case you want to take a look without having to run the script again.



