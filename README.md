# Classification of Rice Leaf Blast Disease Using VGG16 Feature Extraction with Partial Least Squares Regression and eXtreme Gradient Boosting

The repository contains the code and supporting documentation for our undergraduate thesis study **"Classification of Rice Leaf Blast Disease Using VGG16 Feature Extraction with Partial Least Squares Regression and eXtreme Gradient Boosting"** (UST A.Y. 2026).

We evaluated deep feature extraction using a pre-trained **VGG16** backbone paired with custom classifiers, comparing a standard Dense baseline against secondary machine learning models (**VGG16 + PLSR** and **VGG16 + XGBoost**). Our pipeline includes automated image preprocessing (contrast enhancement via CLAHE), dataset balancing and augmentation, and visual model explainability via Grad-CAM.

---

## 📄 Reference Files

* **`DATSCI20_Journal.pdf`**: Research journal in ACM format.
* **`DATSCI20_Documentation.pdf`**: Our complete model performance documentation and logs.
* **`DATSCI20_ResearchDescription.pdf`**: Detailed background and project scope.
* **`DATSCI20_Tarp.png`**: Presentation poster summarizing key findings.

---

## 🛠️ What We Implemented

* **Feature Extraction & Classification Pipeline**: We used a pre-trained VGG16 backbone to extract deep features and benchmarked performance using standard Dense layers, PLSR, and XGBoost.
* **Preprocessing Pipeline**: We implemented CLAHE contrast enhancement alongside dataset balancing and augmentation procedures.
* **Explainability & Testing**: We integrated Grad-CAM heatmaps to inspect model decisions and tested generalization across local and multi-source field datasets.

---

## 📂 Repository Structure

```text
Thesis-DATASCI20-Output/
├── DATSCI20_Codes/
│   ├── Formula Code/                      # Model evaluation scripts (Shapiro-Wilk for normality, paired t-test if parametric, Wilcoxon rank-sum if non-parametric)
│   ├── GUI/                               # Application deployment and demo interfaces (may be incomplete)
│   ├── [0 Prototype] Old Quadriga Archi/  # Early architecture iterations
│   ├── [1 Ver 5] Feature Extraction Ver/  # Feature extraction pipeline version
│   ├── [2 Ver 1] With Processing Image/   # Pipeline integration with image preprocessing
│   ├── [3 Ver 4] CLEHA Image Processing/  # CLAHE enhancement and image processing improvements
│   └── [4 Defense Revision]/              # Final scripts updated post-defense
├── DATSCI20 - Documentation.pdf           # Comprehensive project documentation
├── DATSCI20_Journal.pdf                   # Final journal paper
├── DATSCI20_ResearchDescription.pdf       # Detailed research overview
├── DATSCI20_Tarp.png                      # Project presentation poster
└── README.md                              # Main repository documentation

```

## 👥 Authors & Acknowledgments

* **Authors:** Charles Fredric Inventado, Ken David Pates, John Vincent (JV) Rodelas, and James Vincent Valles
* **Affiliation:** DATASCI20, A.Y. 2026 Graduates, Department of Computer Science, University of Santo Tomas
