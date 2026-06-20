
# Setup 
All experiments were run inside a Docker container, because recent versions of TensorFlow are not
compatible with Windows. You can install the container by running the following command:
```
docker run -it --gpus all -p 8888:8888 -v “$(pwd)/bridge:/tf/bridge” tensorflow/tensorflow:2.10.0-gpu-jupyter
```

# User Guide
The idea is to place the `assignment_3.py` script in a folder along with the `best_model.keras` and `class_names.pkl` files.
The `DATASET` folder is optional.
```
main_dir:
|
|--DATASET (folder)
|    |
|    |--AID (folder)
|    |--UC Merced (folder)
|
|--assignment_3.py
|--best_model.keras
|--class_names.pkl
```
- The `assignment_3.py` script allows you to set flags in case you want to start training from scratch. 
Setting all flags to `False` will skip directly to the inference phase.

- The `assignment_3.py` script contains a series of modular functions that can be used independently. 
- You can read the docstring to understand what each function does, or use the `help` method: `help(method)`.

---

## Example of Use
### Single Use
Simply enter the path to an image, then call the `single_img_classification` method.
```
img_path = “punta_raisi.png”
label, all_probs, class_names = single_img_classification(img_path)
print(f“\n<- Classifying {os.path.basename(img_path)} ->”)
print(f“<- The provided image belongs to the class: {label} ->”)
plot_probs(all_probs, class_names)
```

### Complete Usage
Below are the steps performed to obtain the stated results.
```
# Initialize GPU
setup_gpu()

# FIRST STRATEGY #
pre_training_phase = True
# PRETRAINING PHASE
if pre_training_phase:
    pre_training(train=True, make_plots=True)
finetuning_phase = True
# FINETUNING PHASE
if finetuning_phase:
    finetuning_step(train=True, make_plots=True)

# SECOND STRATEGY #
second_strategy_phase = True
if second_strategy_phase:
    second_strategy(train=True, make_plots=True)

# MODEL SELECTION #
model_selection_phase = True
if model_selection_phase:
    model_selection()
# INFERENCE
img_path = “punta_raisi.png”
label, all_probs, class_names = single_img_classification(img_path)
print(f“\n<- Classifying {os.path.basename(img_path)} ->”)
print(f“<- The provided image belongs to the class: {label} ->”)
plot_probs(all_probs, class_names)
```

---
#### Required libraries:
Python version _3.8.10_.
- matplotlib
- numpy
- opencv-python
- opencv-python-headless 
- pandas==2.0.3 
- scikit-learn==1.3.2 
- split-folders 
- tensorboard==2.10.0 
- tensorboard-data-server==0.6.1 
- tensorboard-plugin-wit==1.8.1 
- tensorflow==2.10.0 
- tensorflow-estimator==2.10.0 
- tensorflow-io-gcs-filesystem==0.26.0
---
#### Notes
The `best_model.keras` file contains the best pre-trained model. 
The `class_names.pkl` file contains a list of all classes in the UC dataset.

### Reproducibility
To ensure full reproducibility of the experiments, a random seed with a value of 177 has been set.
Specifically, this seed is used in the following functions:
- `train_val_test_split`: to ensure that the stratified splits of the datasets into train/val/test are always the same. 
- `load_train_val_sets`: for the random shuffling of images during model training; note that the shuffle is applied only to the train set.