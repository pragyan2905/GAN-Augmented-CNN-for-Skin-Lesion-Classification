# Skin Cancer Detection using Deep Learning 🔬

This project uses a deep learning model to classify different types of skin lesions from images. The goal is to assist in the early detection of skin cancer by accurately identifying various skin conditions.

## Dataset 📊

The model is trained on the **"Skin Cancer ISIC (The International Skin Imaging Collaboration)"** dataset. This dataset contains images categorized into nine distinct classes of skin lesions:

* Actinic Keratosis
* Basal Cell Carcinoma
* Dermatofibroma
* Melanoma
* Nevus
* Pigmented Benign Keratosis
* Seborrheic Keratosis
* Squamous Cell Carcinoma
* Vascular Lesion

The dataset is split into training and validation sets to build and evaluate the model.

## Model Architecture 🧠

This project utilizes a transfer learning approach with the **EfficientNetB0** model, pre-trained on the ImageNet dataset. The architecture is as follows:

1.  **Base Model:** An `EfficientNetB0` model with its top (classification) layer removed. The weights of this base model are initially frozen.
2.  **Flatten Layer:** Flattens the output from the base model into a one-dimensional vector.
3.  **Dense Layer:** A fully connected layer with 128 neurons and a `ReLU` activation function.
4.  **Dropout Layer:** A dropout layer with a rate of 0.5 to prevent overfitting.
5.  **Output Layer:** A final dense layer with a `softmax` activation function to output the probabilities for each of the 9 classes.

## How to Use 🚀

To run this project, follow these steps:

1.  **Set up Kaggle API:**
    * Create a `kaggle.json` file with your Kaggle API credentials.
    * Place this file in the `~/.kaggle/` directory.

2.  **Download the Dataset:**
    * Run the notebook cells that use the Kaggle API to download and unzip the "skin-cancer-detection" dataset.

3.  **Train the Model:**
    * Execute the cells to define the model architecture.
    * The model is trained in two phases:
        1.  Initial training with the base model's layers frozen.
        2.  Fine-tuning where the base model is unfrozen and trained with a lower learning rate.

4.  **Evaluate the Model:**
    * The notebook includes steps to evaluate the model's performance on the validation set.

## Dependencies 💻

This project requires the following Python libraries:

* TensorFlow
* NumPy
* Matplotlib
* Kaggle

You can install them using pip:
```bash
pip install tensorflow numpy matplotlib kaggle

