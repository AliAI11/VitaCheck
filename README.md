# VitaCheck: Symptom-Based Vitamin Deficiency Detection & Personalized Meal Planning

**Authors:** Afeef Ali, Sanjana Gollu, Aarav Bafna, Natalie Rogers, Tiffany Widjaja  
**Course:** Virginia Tech HCI Capstone  
**Date:** April 2026

---

## Description

VitaCheck is an end-to-end machine learning pipeline that classifies vitamin deficiency type from patient-reported symptoms and demographics, then generates a personalized meal plan using a large language model. Users interact with a Gradio web interface where they enter demographics, lifestyle factors, and symptoms.

The system predicts one of five conditions — **Anemia**, **Night Blindness**, **Rickets/Osteomalacia**, **Scurvy**, or **Healthy** — and generates a 3-day meal plan tailored to the predicted deficiency and the user's dietary preferences.

The ML pipeline uses XGBoost trained on 4,000 patient records with SMOTE-NC for class imbalance handling and Optuna for hyperparameter tuning. Meal plans are generated via Groq's free API using Llama 3.3 70B.

> **Best model:** XGBoost (Tuned) — 84.1% accuracy, 0.70 macro F1

---

## Package Contents

```
VitaCheck/
├── README.md                            This file
├── DOC/
│   ├── Project_Milestone.pdf            Milestone report
│   ├── Project_Proposal.pdf             Original proposal
│   └── Capstone_Poster.pdf             Final poster
└── SRC/
    ├── app.py                           Gradio app (for Hugging Face Spaces)
    ├── requirements.txt                 Python dependencies for deployment
    ├── notebooks/
    │   └── HCICapstone.ipynb            Full training notebook (run on Colab)
    └── artifacts/
        ├── xgb_tuned.joblib             Trained XGBoost model
        ├── scaler.joblib                Fitted StandardScaler
        ├── target_label_encoder.joblib  Fitted LabelEncoder
        └── feature_info.json           Feature metadata
```

---

## Installation

**Requirements:**
- Python 3.10+
- pip

**Install dependencies:**
```bash
pip install -r SRC/requirements.txt
```

**For training (optional — pretrained model included):**
- Google Colab with A100 GPU is recommended
- Upload `SRC/notebooks/HCICapstone.ipynb` to Colab and run all cells
- Note: full training with Optuna tuning takes approximately 60 minutes

**For the meal plan feature:**
- Create a free Groq API key at https://console.groq.com/keys
- Set it as an environment variable:
```bash
export GROQ_API_KEY="gsk_..."
```

---

## Usage — Run Locally

1. Navigate to the `SRC` directory:
   ```bash
   cd SRC
   ```

2. Run the Gradio app:
   ```bash
   python app.py
   ```

3. Open the local URL printed in the terminal (typically `http://127.0.0.1:7860`).

4. Enter demographics, lifestyle factors, and symptoms, then click **"Analyze & Generate Meal Plan"**.

5. View results on the **"Diagnosis"** tab and the generated plan on the **"Personalized Meal Plan"** tab.

---

## Usage — Hugging Face Spaces (Live Demo)

The app is deployed at:  
**https://huggingface.co/spaces/aali11/VitaCheck**

No installation needed — just visit the link and use the interface.

---

## Dataset

**Vitamin Deficiency Disease Prediction Dataset** (Kaggle, January 2026)
- 4,000 patient records, 34 columns
- 5 classes: Healthy, Anemia, Rickets/Osteomalacia, Night Blindness, Scurvy
- Available at: https://www.kaggle.com/datasets/nudratabbas/vitamin-deficiency-disease-prediction-dataset

The dataset is **not** included in this package to keep the file size small. To retrain, the notebook downloads it automatically via `kagglehub`.

---

## Technologies

| Component | Technology |
|---|---|
| Classification | XGBoost, LightGBM, scikit-learn |
| Class imbalance handling | SMOTE-NC |
| Hyperparameter tuning | Optuna |
| Model explainability | SHAP |
| Meal plan generation | Groq API + Llama 3.3 70B |
| Web interface | Gradio |
| Deployment | Hugging Face Spaces |
