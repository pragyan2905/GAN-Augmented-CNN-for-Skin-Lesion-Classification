# Skin Cancer Detection using Deep Learning 🧠

## Overview 🔬

This project focuses on the classification of skin lesions from images to detect various types of skin cancer. We employ a deep learning model, specifically a **Convolutional Neural Network (CNN)**, to accurately categorize images into nine distinct classes of skin lesions, including melanoma, basal cell carcinoma, and others. The primary goal is to leverage advanced computer vision techniques to assist in the early detection of skin cancer.

---

## Dataset 📁

The model is trained on the "Skin cancer ISIC The International Skin Imaging Collaboration" dataset, which is available on Kaggle. This dataset contains thousands of images categorized into nine different classes:

- actinic keratosis
- basal cell carcinoma
- dermatofibroma
- melanoma
- nevus
- pigmented benign keratosis
- seborrheic keratosis
- squamous cell carcinoma
- vascular lesion

---

## Methodology ⚙️

To achieve high accuracy, this project utilizes a state-of-the-art approach combining several powerful techniques:

### 1. Transfer Learning 🚀

Instead of training a **CNN** from scratch, we use **transfer learning**. This technique involves using a pre-trained model that has already learned features from a massive dataset (ImageNet). For this project, we use the **EfficientNetB0** architecture as our base model. This allows us to leverage the powerful feature extraction capabilities of a well-established model, saving significant training time and computational resources.

### 2. Data Augmentation ✨

To prevent overfitting and improve the model's ability to generalize to new, unseen images, we apply **data augmentation**. This involves creating modified versions of the existing training images by applying random transformations, such as:

-   Horizontal flipping
-   Random rotations
-   Random zooming

This process artificially expands the training dataset, exposing the model to a wider variety of image variations.

### 3. Fine-Tuning 🔧

The training process is divided into two main phases:

1.  **Initial Training**: First, we freeze the layers of the pre-trained EfficientNetB0 model and train only the newly added classification layers. This allows the new layers to adapt to the specifics of our skin lesion dataset without disrupting the learned weights of the base model.
2.  **Fine-Tuning**: After the initial training, we unfreeze the entire model and continue training with a very low learning rate. This **fine-tuning** step allows the whole network, including the base model's layers, to make small adjustments to its weights, further specializing it for the task of skin cancer classification.

---

## How to Use 📖

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-link>
    ```
2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy matplotlib
    ```
3.  **Set up Kaggle API:**
    -   Make sure you have your `kaggle.json` file in the appropriate directory (`~/.kaggle/`).
4.  **Run the Jupyter Notebook:**
    -   Open and run the `skin_cancer_detection.ipynb` notebook to download the data, train the model, and evaluate its performance.

---

## Results 📊

The model is trained to classify the nine different types of skin lesions. The use of **transfer learning**, **data augmentation**, and **fine-tuning** contributes to a robust and accurate classification model. The training history, including accuracy and loss metrics, can be visualized within the notebook to assess the model's performance over epochs.
