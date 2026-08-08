<div align="center">

# 🌾 Classification of Rice Leaf Blast Disease
### Using VGG16 Feature Extraction with Partial Least Squares Regression & eXtreme Gradient Boosting

![Degree](https://img.shields.io/badge/Degree-B.S._Computer_Science-blue?style=flat-square)
![Institution](https://img.shields.io/badge/Institution-University_of_Santo_Tomas-gold?style=flat-square)
![Academic Year](https://img.shields.io/badge/Academic_Year-2025_--_2026-green?style=flat-square)

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Classifier-111111?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-CLAHE-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)

</div>

The repository contains the code and supporting documentation for our undergraduate thesis study **"Classification of Rice Leaf Blast Disease Using VGG16 Feature Extraction with Partial Least Squares Regression and eXtreme Gradient Boosting"** (UST A.Y. 2026).

We evaluated deep feature extraction using a pre-trained **VGG16** backbone paired with custom classifiers, comparing a standard Dense baseline against secondary machine learning models (**VGG16 + PLSR** and **VGG16 + XGBoost**). Our pipeline includes automated image preprocessing (contrast enhancement via CLAHE), dataset balancing and augmentation, and visual model explainability via Grad-CAM.

---

## 🎯 Problem Statement & Key Conclusions

> ### ⚠️ The Problem
> Rice leaf blast (*Magnaporthe oryzae*) poses a severe threat to global rice yield, requiring rapid, accurate, and automated disease detection. While deep learning Convolutional Neural Networks (CNNs) like VGG16 are effective feature extractors, standard dense classification layers often suffer from high computational overhead, overfitting, and reduced generalization when deployed across varied field environments—such as shifting from multi-source global benchmark datasets to real-world local crops.

### 📈 Key Conclusions & Statistical Results

| Metric / Phase | Baseline VGG16 | VGG16 + XGBoost | VGG16 + PLSR |
| :--- | :---: | :---: | :---: |
| **Validation Accuracy (Multi-Source)** | **80.13% – 84.29%** | **80.13% – 84.29%** | **80.13% – 84.29%** |
| **Local Field Test Accuracy (Zambali)** | **76.44%** 🏆 | 55.05% | 54.09% |
| **Recall / Sensitivity** | **94.86%** | 69.71% | 55.43% |
| **F1-Score** | **77.21%** | 56.61% | 50.39% |
| **Inference Time** | 1.36s | **0.07s** ⚡ | **0.06s** ⚡ |

* **Validation Performance:** When trained and validated on the Multi-Source Rice Leafs dataset, all models achieved strong accuracy ranging between **80.13%** and **84.29%**.
* **Cross-Domain Generalization (Local Zambali Field Test):** When tested on the local Zambali dataset with augmented data, the **Baseline VGG16** achieved the highest classification accuracy of **76.44%** (Recall: **94.86%**, F1-score: **77.21%**), outperforming the hybrid models.
* **Impact of Data Augmentation:** Data augmentation significantly improved the Baseline VGG16 model's local test accuracy from **52.40%** (without augmentation) to **76.44%** (with augmentation), though its benefits were limited for the hybrid architectures.
* **Explainability Insights:** Grad-CAM visual heatmaps revealed that the Baseline VGG16 successfully focused on leaf lesions, whereas the hybrid models frequently focused on background noise, leading to higher false-positive rates.
* **Statistical Significance:** Statistical testing using the **Paired T-Test** showed no statistically significant difference in performance between the hybrid models and the baseline, indicating that the added architectural complexity did not yield a performance advantage.

---

## 📄 Reference Files

| Document | Format | Description | Link / Location |
| :--- | :---: | :--- | :--- |
| **Thesis Manuscript** | `PDF` | Full Undergraduate Thesis Document | [📥 View on Google Drive](https://drive.google.com/file/d/1g0qs2jMMuvtVUh_RYSbjl1XyBXcjZCBK/view?usp=drive_link) |
| **ACM Research Journal** | `PDF` | Research journal formatted in ACM style | [`DATSCI20_Journal.pdf`](./DATSCI20_Journal.pdf) |
| **Model Documentation** | `PDF` | Complete performance logs & evaluations | [`DATSCI20_Documentation.pdf`](./DATSCI20_Documentation.pdf) |
| **Research Description** | `PDF` | Detailed background, scope, & objectives | [`DATSCI20_ResearchDescription.pdf`](./DATSCI20_ResearchDescription.pdf) |
| **Presentation Poster** | `PNG` | High-res tarp display summarizing key findings | [`DATSCI20_Tarp.png`](./DATSCI20_Tarp.png) |

---

## 📊 Datasets

Our project evaluates cross-domain performance using two primary datasets, focusing specifically on the **Healthy (H)** and **Leaf Blast (LB)** classes:

> 📁 **Project Data Hub:** Access both processed and formatted datasets via our [Google Drive Folder](https://drive.google.com/drive/folders/1bW5pkNVCEGWXOjS6-q1Eqxo2-ynxB6dp?usp=sharing).

### Dataset Breakdown

<details>
<summary><b>1. Multi-Source Rice Leaf Dataset (Training & Validation)</b></summary>

* **Source:** [Kaggle - Rice Leaf Dataset](https://www.kaggle.com/datasets/shayanriyaz/riceleafs)
* **Raw Class Counts:** 1,488 Healthy (H) images & 779 Leaf Blast (LB) images.
* **Usage:** Base dataset for model training, feature extraction, and validation.
</details>

<details>
<summary><b>2. Zambali Rice Dataset (Cross-Domain Testing)</b></summary>

* **Source:** [Kaggle - Zambali Rice Dataset v3.1](https://www.kaggle.com/datasets/gettingintoml/zambali-rice-dataset-v3-1)
* **Raw Class Counts:** 309 Healthy (H) images & 212 Leaf Blast (LB) images.
* **Usage:** Cross-domain deployment testing to evaluate model generalization on local Philippine field images gathered from Zambales.
</details>

### 🏷️ File Standardizing & Naming Convention
Both datasets underwent file name standardization during preprocessing to resolve incorrect labeling:

```text
[TYPE]_[NUMBER]_[DATASET]_[MODIFICATION]
│       │        │         └── Processing state (e.g., CLAHE, augmented)
│       │        └────────── 'foreignI' (Multi-Source) or 'LOCAL' (Zambali)
│       └────────────────── Sequential numeric ID (e.g., 1, 2, 55...)
└───────────────────────── Class label: 'HEALTHY' or 'LEAFBLAST'
```
## 👥 Authors & Acknowledgments

* **Authors:** Charles Fredric Inventado, Ken David Pates, John Vincent (JV) Rodelas, and James Vincent Valles
* **Affiliation:** DATASCI20, A.Y. 2026 Graduates, Department of Computer Science, University of Santo Tomas
* **Thesis Adviser:** Mr. Ahdrian Camilo Gernale
* **Thesis Coordinator:** Prof. Donata Acula
