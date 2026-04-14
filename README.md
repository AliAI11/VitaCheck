VitaCheck: Symptom-Based Vitamin Deficiency Detection & Personalized Meal Planning

Authors: Afeef Ali, Sanjana Gollu, Aarav Bafna, Natalie Rogers, Tiffany Widjaja
Course:  Virginia Tech HCI Capstone
Date:    April 2026

DESCRIPTION

VitaCheck is an end-to-end machine learning pipeline that classifies vitamin
deficiency type from patient-reported symptoms and demographics, then generates
a personalized meal plan using a large language model. Users interact with a
Gradio web interface where they enter demographics, lifestyle factors, and
symptoms. The system predicts one of five conditions (Anemia, Night Blindness,
Rickets/Osteomalacia, Scurvy, or Healthy) and generates a 3-day meal plan
tailored to the predicted deficiency and the user's dietary preferences.

The ML pipeline uses XGBoost trained on 4,000 patient records with SMOTE-NC
for class imbalance handling and Optuna for hyperparameter tuning. Meal plans
are generated via Groq's free API using Llama 3.3 70B.

Best model: XGBoost (Tuned) — 84.1% accuracy, 0.70 macro F1

PACKAGE CONTENTS

VitaCheck/
├── README.md                   - this file
├── DOC/
│   ├── Project_Milestone.pdf   - milestone report
│   ├── Project_Proposal.pdf    - original proposal
│   └── Capstone_Poster.pdf     - final poster 
├── SRC/
│   ├── app.py                  - gradio app (for huggingface spaces)
│   ├── requirements.txt        - python dependencies for deployment
│   ├── notebooks/
│   │   └── HCICapstone.ipynb   - full training notebook (run on colab)
│   └── artifacts/
│       ├── xgb_tuned.joblib            - trained xgboost model
│       ├── scaler.joblib               - fitted standardscaler
│       ├── target_label_encoder.joblib - fitted labelencoder
│       └── feature_info.json           - feature metadata

INSTALLATION

requirements:
- python 3.10+
- pip

install dependencies:

    pip install -r SRC/requirements.txt

for training (optional — pretrained model included):
- google colab with A100 GPU recommended
- upload SRC/notebooks/HCICapstone.ipynb to colab
- run all cells (takes ~60 min with optuna tuning)

for the meal plan feature:
- create a free groq api key at https://console.groq.com/keys
- set it as environment variable: export GROQ_API_KEY="gsk_..."

USAGE — RUN LOCALLY

1. navigate to the SRC directory:

    cd SRC

2. run the gradio app:

    python app.py

3. open the local url printed in terminal (typically http://127.0.0.1:7860)

4. enter demographics, lifestyle, and symptoms, then click
   "analyze & generate meal plan"

5. view diagnosis on the "diagnosis" tab and meal plan on the
   "personalized meal plan" tab

USAGE — HUGGINGFACE SPACES (LIVE DEMO)

the app is deployed at:

    [https://huggingface.co/spaces/AliAI11/VitaCheck](https://huggingface.co/spaces/aali11/VitaCheck)

no installation needed — just visit the link and use the interface.

DATASET

vitamin deficiency disease prediction dataset (kaggle, january 2026)
- 4,000 patient records, 34 columns
- 5 classes: healthy, anemia, rickets/osteomalacia, night blindness, scurvy
- available at: kaggle.com/datasets/nudratabbas/vitamin-deficiency-disease-prediction-dataset

the dataset is NOT included in this package to keep the file size small.
to retrain, the notebook downloads it automatically via kagglehub.

TECHNOLOGIES

- xgboost + lightgbm + scikit-learn (classification)
- smote-nc (class imbalance handling)
- optuna (hyperparameter tuning)
- shap (model explainability)
- groq api + llama 3.3 70b (meal plan generation)
- gradio (web interface)
- huggingface spaces (deployment)

================================================================================
