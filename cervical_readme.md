# 🔬 Cervical Cancer Cell Classification: AI-Powered Pap Smear Analysis

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.2-red.svg)](https://pytorch.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An advanced deep learning system for automated classification of cervical cells from Pap smear images using EfficientNet-B0 architecture. This project aims to assist healthcare professionals in early detection of cervical abnormalities and precancerous conditions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [About Cervical Cancer](#about-cervical-cancer)
- [Cell Classification Types](#cell-classification-types)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Streamlit Web Application](#streamlit-web-application)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

Cervical cancer is one of the most preventable cancers when detected early through regular screening. This project leverages state-of-the-art deep learning techniques to classify cervical cells into five distinct categories, helping cytotechnologists and pathologists in their diagnostic workflow.

**Key Features:**
- ✅ Multi-class classification of 5 cervical cell types
- ✅ EfficientNet-B0 backbone for optimal accuracy-efficiency trade-off
- ✅ Interactive Streamlit web application for easy deployment
- ✅ Comprehensive cell type information and risk assessment
- ✅ Visual confidence scores and probability distributions
- ✅ Educational resources about cervical cancer screening

---

## About Cervical Cancer

### What is Cervical Cancer?

Cervical cancer develops in the cells of the cervix — the lower part of the uterus that connects to the vagina. Most cervical cancers are caused by various strains of the human papillomavirus (HPV), a sexually transmitted infection.

### The Importance of Pap Smear Testing

A Pap smear (Pap test) is a screening procedure that:
- Detects precancerous or cancerous cells on the cervix
- Allows early intervention before cells become cancerous
- Has reduced cervical cancer mortality by 60-90% in populations with regular screening
- Is recommended every 3 years for women aged 21-65

### Risk Factors

- **HPV infection** (primary cause)
- Smoking
- Weakened immune system
- Multiple sexual partners
- Early onset of sexual activity
- Long-term oral contraceptive use
- Multiple full-term pregnancies

### Screening Guidelines

| Age Group | Recommended Screening |
|-----------|----------------------|
| 21-29 years | Pap test every 3 years |
| 30-65 years | Pap test every 3 years OR HPV test every 5 years OR Co-testing every 5 years |
| 65+ years | May discontinue if previous tests were normal |

---

## Cell Classification Types

This model classifies cervical cells into five distinct categories:

### 1. 🔴 Dyskeratotic
**Risk Level:** High

**Description:** Cells showing abnormal keratinization, often indicating pre-cancerous changes.

**Clinical Significance:** May indicate cervical intraepithelial neoplasia (CIN) or dysplasia.

**Key Characteristics:**
- Dense cytoplasm
- Irregular nucleus
- Premature keratinization
- Associated with HPV infection

**Recommended Action:** Further cytological evaluation, consider colposcopy and biopsy.

---

### 2. 🟠 Koilocytotic
**Risk Level:** Moderate to High

**Description:** Cells exhibiting features of HPV infection with characteristic perinuclear clearing.

**Clinical Significance:** Direct indicator of Human Papillomavirus (HPV) infection.

**Key Characteristics:**
- Perinuclear halo (clearing around nucleus)
- Enlarged, irregular nucleus
- Binucleation or multinucleation
- Irregular nuclear membrane

**Recommended Action:** HPV testing, follow-up Pap smear, possible colposcopy.

---

### 3. 🟡 Metaplastic
**Risk Level:** Low

**Description:** Cells undergoing normal transformation in the transformation zone of the cervix.

**Clinical Significance:** Usually represents normal cellular adaptation, but requires monitoring.

**Key Characteristics:**
- Immature squamous cells
- May show reactive changes
- Variable cytoplasm
- Round to oval nuclei

**Recommended Action:** Routine follow-up, monitor for progression.

---

### 4. 🟡 Parabasal
**Risk Level:** Low to Moderate

**Description:** Immature cells from the basal layer of the epithelium.

**Clinical Significance:** May indicate atrophy, inflammation, or hormonal changes.

**Key Characteristics:**
- Small cell size
- High nuclear-to-cytoplasmic ratio
- Dense chromatin
- Often seen in post-menopausal women or inflammation

**Recommended Action:** Clinical correlation with patient history, consider hormonal status.

---

### 5. 🟢 Superficial-Intermediate
**Risk Level:** Normal

**Description:** Mature squamous cells from the upper layers of the cervical epithelium.

**Clinical Significance:** Represents healthy, normal cervical epithelium.

**Key Characteristics:**
- Large, flat cells
- Small, regular nuclei
- Abundant cytoplasm
- Well-defined cell borders

**Recommended Action:** Continue routine screening schedule.

---

## Dataset

### SIPaKMeD Dataset

This project utilizes the **SIPaKMeD (Single Cell Images for Pap Smear Analysis)** dataset, a comprehensive collection of cervical cell images.

**Dataset Specifications:**
- **Total Images:** 4,049 isolated cell images
- **Image Size:** 224 × 224 pixels
- **Classes:** 5 cell types
- **Source:** Medical images from Pap smear tests
- **Format:** BMP images

**Class Distribution:**
| Cell Type | Number of Images |
|-----------|-----------------|
| Superficial-Intermediate | 831 |
| Parabasal | 787 |
| Koilocytotic | 825 |
| Dyskeratotic | 813 |
| Metaplastic | 793 |

**Data Preparation:**
- Images are preprocessed and normalized
- Train/Validation/Test split: 70/15/15
- Augmentation techniques applied during training (rotation, flip, zoom)

### Downloading the Dataset

You can download the SIPaKMeD dataset from Kaggle:

```bash
# Using Kaggle API
kaggle datasets download -d prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed

# Or manually from:
# https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed
```

---

## Model Architecture

### EfficientNet-B0

This project employs **EfficientNet-B0**, a state-of-the-art CNN that achieves excellent accuracy with computational efficiency.

**Architecture Highlights:**
- **Backbone:** EfficientNet-B0 (pre-trained on ImageNet)
- **Input Size:** 224 × 224 × 3
- **Output Classes:** 5 (cell types)
- **Parameters:** ~5.3M
- **Activation:** Softmax for multi-class classification

**Key Features:**
- Compound scaling method balancing depth, width, and resolution
- Mobile inverted bottleneck convolution (MBConv)
- Squeeze-and-excitation blocks for channel attention
- Efficient parameter utilization

---

## Project Structure

```
📂 Cervical_cancer_detection_model/
│
├── 📄 README.md                          # Project documentation
├── 📄 pyproject.toml                     # Python project dependencies
├── 📄 uv.lock                            # Dependency lock file
├── 📄 .python-version                    # Python version specification
├── 📄 .gitignore                         # Git ignore rules
├── 📄 Dockerfile                         # Docker containerization
│
├── 📓 cervical-model.ipynb               # Jupyter notebook for training & analysis
│
├── 📂 app/                               # Flask application (if applicable)
│   └── ...                               # API endpoints and serving logic
│
├── 📂 models/                            # Trained model files
│   └── efficientnet_b0_cervical_3way_split.pth
│
├── 📂 streamlit/                         # Streamlit web application
│   ├── 📄 app.py                         # Main Streamlit application
│   ├── 📄 requirements.txt               # Streamlit dependencies
│   └── 📂 models/                        # Model files for Streamlit
│
└── 📂 data/                              # Dataset directory (not tracked)
    ├── train/
    ├── val/
    └── test/
```

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- CUDA-compatible GPU (optional, for faster training)
- 8GB+ RAM recommended

### 1. Clone the Repository

```bash
git clone https://github.com/lanarkite99/Cervical_cancer_detection_model.git
cd Cervical_cancer_detection_model
```

### 2. Create Virtual Environment

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n cervical python=3.11
conda activate cervical
```

### 3. Install Dependencies

```bash
# Install PyTorch (choose appropriate version for your system)
pip install torch==2.2.2 torchvision==0.17.2

# Install other requirements
pip install -r streamlit/requirements.txt
```

### 4. Download Dataset & Model

```bash
# Download dataset (see Dataset section above)
# Place in data/ directory

# Download pre-trained model
# Model file: efficientnet_b0_cervical_3way_split.pth
# Place in models/ directory
```

---

## Usage

### Training the Model

To train the model from scratch:

```bash
# Open Jupyter notebook
jupyter notebook cervical-model.ipynb

# Or run training script (if available)
python train.py --epochs 50 --batch-size 32
```

### Running the Streamlit Application

```bash
cd streamlit
streamlit run app.py
```

The application will be available at `http://localhost:8501`

### Using Docker

```bash
# Build Docker image
docker build -t cervical-cancer-classifier .

# Run container
docker run -p 8501:8501 cervical-cancer-classifier
```

### Making Predictions (Python API)

```python
import torch
from torchvision import transforms
from PIL import Image

# Load model
model = torch.load('models/efficientnet_b0_cervical_3way_split.pth')
model.eval()

# Preprocess image
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load and predict
image = Image.open('path/to/cell_image.jpg').convert('RGB')
tensor = preprocess(image).unsqueeze(0)

with torch.no_grad():
    output = model(tensor)
    probabilities = torch.softmax(output, dim=1)
    predicted_class = torch.argmax(probabilities, dim=1)

print(f"Predicted class: {CLASSES[predicted_class.item()]}")
print(f"Confidence: {probabilities[0][predicted_class].item():.2%}")
```

---

## Model Performance

### Classification Metrics

| Metric | Score |
|--------|-------|
| **Overall Accuracy** | 93.59% |
| **Precision (Macro Avg)** | 93.62% |
| **Recall (Macro Avg)** | 93.59% |
| **F1-Score (Macro Avg)** | 93.60% |

### Confusion Matrix

The model shows strong performance across all cell types with minimal confusion between classes.

---

## Streamlit Web Application

### Features

✨ **User-Friendly Interface**
- Clean, professional medical-themed design
- Drag-and-drop image upload
- Real-time prediction with confidence scores

📊 **Comprehensive Analysis**
- Interactive probability charts using Plotly
- Detailed cell type information
- Risk assessment and clinical recommendations

📚 **Educational Content**
- Complete guide to cervical cancer screening
- Information about all 5 cell types
- Screening guidelines by age group

🔒 **Medical Safety**
- Clear disclaimer about tool limitations
- Emphasis on professional review
- Guidelines for follow-up actions

### Screenshot

![Streamlit App Interface](D:\Cervical_cancer\Capture.PNG)

### Deployment

The application can be deployed on:
- Streamlit Cloud (easiest & current)
- Heroku
- AWS/GCP/Azure
- Docker container on any cloud platform

---

## Future Improvements

### Planned Features

- [ ] **Multi-model Ensemble:** Combine predictions from multiple architectures
- [ ] **Attention Visualization:** Highlight regions model focuses on
- [ ] **Batch Processing:** Analyze multiple images simultaneously
- [ ] **PDF Report Generation:** Export detailed analysis reports
- [ ] **API Integration:** REST API for programmatic access
- [ ] **Mobile Application:** iOS/Android app for point-of-care use

### Research Directions

- [ ] **Explainable AI:** Implement GRAD-CAM for interpretability
- [ ] **Semi-supervised Learning:** Leverage unlabeled data
- [ ] **Active Learning:** Improve model with challenging cases
- [ ] **Multi-task Learning:** Joint segmentation and classification
- [ ] **Few-shot Learning:** Adapt to rare cell types

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Guidelines

- Follow PEP 8 style guide for Python code
- Add tests for new features
- Update documentation accordingly
- Ensure all tests pass before submitting PR

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

### Dataset
- **SIPaKMeD Dataset:** Special thanks to the creators of the SIPaKMeD dataset for making this research possible
- Available on Kaggle: [Cervical Cancer Dataset](https://www.kaggle.com/datasets/prahladmehandiratta/cervical-cancer-largest-dataset-sipakmed)

### Frameworks & Libraries
- **PyTorch:** For the deep learning framework
- **Streamlit:** For the interactive web application
- **EfficientNet:** For the model architecture
- **Plotly:** For interactive visualizations

### Inspiration
- Research papers on automated cervical cancer screening
- Medical professionals working in cytopathology
- Open-source medical AI community

### References

1. Plissiti, M. E., et al. (2018). "SIPaKMeD: A New Dataset for Feature and Image Based Classification of Normal and Pathological Cervical Cells in Pap Smear Images."
2. Tan, M., & Le, Q. (2019). "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks."
3. WHO Guidelines for Screening and Treatment of Cervical Pre-cancer Lesions

---

## Contact & Support

**Project Maintainer:** [lanarkite99](https://github.com/lanarkite99)

**Disclaimer:** This tool is for research and educational purposes only. It is not intended to replace professional medical diagnosis. Always consult qualified healthcare professionals for medical decisions.


---

<div align="center">

**Made with ❤️ for advancing cervical cancer screening through AI**

⭐ Star this repository if you find it helpful!

</div>