# GAN-Augmented Convolutional Neural Network for Skin Lesion Classification

This project leverages Deep Convolutional Generative Adversarial Networks (DCGAN) to augment an imbalanced medical imaging dataset, significantly improving the performance of a CNN classifier for skin lesion detection.

## 📋 Table of Contents

- [Overview](#overview)
- [The Problem: Class Imbalance](#the-problem-class-imbalance)
- [Our Solution: GAN-Based Augmentation](#our-solution-gan-based-augmentation)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
- [Results](#results)
- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

Medical imaging datasets often suffer from severe class imbalance, where common conditions vastly outnumber rare but critical diseases. This project addresses this challenge by using GANs to generate synthetic images of minority classes, creating a balanced dataset that enables more reliable disease classification.

## ❗ The Problem: Class Imbalance

Standard machine learning models struggle with imbalanced medical imaging datasets. The HAM10000 dermatology dataset exemplifies this issue:

- **nv** (Benign Moles): 6,705 images
- **mel** (Melanoma): ~1,113 images
- **bkl** (Benign Keratosis): ~1,099 images
- **bcc** (Basal Cell Carcinoma): ~514 images
- **akiec** (Actinic Keratoses): ~327 images
- **vasc** (Vascular lesions): 142 images
- **df** (Dermatofibroma): 115 images

### The Accuracy Paradox

A classifier trained on this imbalanced data achieves high overall accuracy by specializing in the majority class while performing poorly on minority classes—often the most clinically significant. This renders the model ineffective for real-world diagnostic applications.

## 💡 Our Solution: GAN-Based Augmentation

We implemented a three-phase strategy to overcome class imbalance:

### Phase 1: Baseline Model
Trained a baseline CNN on the original imbalanced dataset to establish performance benchmarks and identify weaknesses in minority class detection.

### Phase 2: DCGAN Training & Synthetic Image Generation
- Built and trained separate DCGANs for each minority class (vasc, df, akiec, bcc, mel)
- Each GAN learned the unique feature distribution of its target class
- Generated hundreds of realistic synthetic images per minority class

### Phase 3: Final Augmented Model
- Combined synthetic images with the original dataset
- Created a balanced training set
- Trained an identical CNN architecture on augmented data
- Achieved substantial improvements in minority class detection

## 🛠 Technology Stack

### Core Technologies
- **Language**: Python 3.x
- **Deep Learning**: TensorFlow, Keras
- **Data Processing**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Machine Learning**: Scikit-learn
- **Dataset Access**: Opendatasets

### Dataset
- **HAM10000**: 10,015 dermatoscopic images across 7 diagnostic categories
- Source: Human Against Machine with 10000 training images

## 🏗 Architecture

### CNN Classifier Architecture

**Convolutional Base:**
- Three convolutional blocks with increasing filter depth (32 → 64 → 128)
- Each block contains:
  - Conv2D layer
  - BatchNormalization
  - ReLU Activation
  - MaxPooling2D
  - Dropout for regularization

**Classifier Head:**
- Flatten layer
- Dense layer (512 units)
- BatchNormalization
- Dropout
- Output Dense layer with softmax activation (7 classes)

### DCGAN Architecture

**Generator:**
- Input: 100-dimensional latent vector
- Architecture: Conv2DTranspose layers with BatchNormalization
- Output: 128×128×3 synthetic image (tanh activation)
- Purpose: Upsample random noise into realistic skin lesion images

**Discriminator:**
- Input: 128×128×3 image (real or synthetic)
- Architecture: Conv2D layers with LeakyReLU and Dropout
- Output: Binary classification (real vs. fake)
- Purpose: Distinguish authentic from generated images

## 📊 Results

### Performance Comparison

| Class | Baseline Recall | Augmented Recall | Improvement |
|-------|----------------|------------------|-------------|
| akiec | 0.59 | 0.88 | **+49%** |
| bcc | 0.59 | 0.91 | **+54%** |
| bkl | 0.25 | 0.79 | **+216%** |
| df | 0.35 | 0.94 | **+168%** |
| mel | 0.43 | 0.85 | **+97%** |
| nv | 0.92 | 0.94 | +2% |
| vasc | 0.22 | 0.96 | **+336%** |
| **Overall Accuracy** | **75%** | **91%** | **+16%** |

### Key Insights
- Dramatic improvement in minority class recall, with some classes showing over 300% enhancement
- The model now reliably identifies rare and malignant conditions
- Maintains high performance on majority classes while becoming clinically useful for all lesion types

## 🚀 Getting Started

### Prerequisites

```bash
pip install tensorflow pandas opendatasets matplotlib seaborn scikit-learn numpy
```

### Kaggle API Setup

1. Create a Kaggle account and generate API credentials
2. Place `kaggle.json` in `~/.kaggle/` (Linux/Mac) or `C:\Users\<Username>\.kaggle\` (Windows)
3. Set permissions: `chmod 600 ~/.kaggle/kaggle.json`

### Installation & Execution

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/gan-skin-lesion-classifier.git
cd gan-skin-lesion-classifier
```

2. **Run the baseline model:**
```bash
jupyter notebook baseline_model.ipynb
```
This notebook will:
- Download the HAM10000 dataset from Kaggle
- Perform exploratory data analysis
- Train and evaluate the baseline CNN

3. **Train GANs and final model:**
```bash
jupyter notebook skin_gan.ipynb
```
This notebook will:
- Initialize the GANTrainer class
- Train separate GANs for minority classes
- Generate synthetic images
- Assemble the balanced dataset
- Train and evaluate the final augmented CNN

## 📁 Project Structure

```
gan-skin-lesion-classifier/
│
├── baseline_model.ipynb          # Baseline CNN training and evaluation
├── skin_gan.ipynb                # GAN training and augmented model
├── data/                          # Dataset directory (auto-created)
│   └── ham10000/                  # Downloaded HAM10000 images
├── generated_images/              # Synthetic images from GANs
├── models/                        # Saved model weights
├── results/                       # Performance metrics and visualizations
└── README.md                      # Project documentation
```

## 🔮 Future Work

- Implement additional augmentation techniques (rotation, flipping, color jittering)
- Experiment with more advanced GAN architectures (StyleGAN, ProGAN)
- Add explainability features (Grad-CAM, attention maps)
- Deploy as a web application for clinical demonstration
- Extend to multi-modal learning with patient metadata
- Validate on external dermatology datasets

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

