# ============================================================================
# vitacheck — full gradio app with groq llm meal plan generation
# ============================================================================
import gradio as gr
import joblib
import pandas as pd
import numpy as np
import json
import os
from groq import Groq

# load artifacts
model = joblib.load('xgb_tuned.joblib')
scaler_obj = joblib.load('scaler.joblib')
target_le = joblib.load('target_label_encoder.joblib')

with open('feature_info.json', 'r') as f:
    info = json.load(f)

feature_columns = info['feature_columns']
class_names = info['class_names']
continuous_cols = info['continuous_cols']

# groq client — key will be set as huggingface space secret
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "(place your api key here)")
groq_client = Groq(api_key=GROQ_API_KEY)


def generate_meal_plan(diagnosis, diet_type, allergies):
    """generate a personalized meal plan using groq llama 3.3"""
    prompt = f"""you are a clinical nutritionist. a patient has been screened and the result is: {diagnosis.replace('_', ' ')}.

patient details:
- diet type: {diet_type}
- allergies/restrictions: {allergies if allergies else 'none reported'}

create a personalized 3-day meal plan to address this condition. for each day provide breakfast, lunch, dinner, and one snack. after the meal plan, add a section called "key nutrients to focus on" explaining which vitamins/minerals to prioritize and why.

keep it practical with real recipes. use markdown formatting. do not use emojis."""

    try:
        response = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": "you are a registered dietitian. give evidence-based, practical dietary advice. always remind patients this is not a substitute for professional medical care. do not use emojis."},
                {"role": "user", "content": prompt}
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.7,
            max_tokens=2048
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"meal plan generation temporarily unavailable. try again in a moment.\n\nerror: {str(e)}"


# nutrient info per diagnosis
nutrient_info = {
    'Anemia': {'deficiency': 'iron, vitamin b12, folate',
               'foods': 'red meat, spinach, lentils, fortified cereals, eggs, dark leafy greens',
               'tip': 'pair iron-rich foods with vitamin c to boost absorption. avoid tea/coffee with meals.'},
    'Night_Blindness': {'deficiency': 'vitamin a',
                        'foods': 'carrots, sweet potatoes, spinach, kale, eggs, liver, mangoes',
                        'tip': 'vitamin a is fat-soluble — eat with healthy fats for better absorption.'},
    'Rickets_Osteomalacia': {'deficiency': 'vitamin d, calcium',
                             'foods': 'salmon, sardines, fortified milk, eggs, uv-exposed mushrooms',
                             'tip': 'aim for 15-20 min sunlight daily. consider a vitamin d supplement.'},
    'Scurvy': {'deficiency': 'vitamin c',
               'foods': 'oranges, strawberries, bell peppers, broccoli, kiwi, guava',
               'tip': 'vitamin c is heat-sensitive — eat fruits/vegetables raw or lightly cooked.'},
    'Healthy': {'deficiency': 'none detected',
                'foods': 'maintain a balanced diet with whole grains, lean proteins, fruits, vegetables',
                'tip': 'keep variety in your diet to cover all micronutrient needs.'}
}


def predict(age, gender, bmi, smoking, alcohol, exercise, diet, sun,
            income, latitude, night_blindness, fatigue, bleeding_gums,
            bone_pain, muscle_weakness, numbness_tingling, memory_problems,
            pale_skin, multiple_deficiencies, allergies):

    alcohol_map = {
        "none": "Unknown",
        "moderate": "Moderate",
        "heavy": "Heavy",
        "prefer not to say": "Unknown"
    }
    alcohol_mapped = alcohol_map.get(alcohol.lower(), "Unknown")

    symptoms = [night_blindness, fatigue, bleeding_gums, bone_pain,
                muscle_weakness, numbness_tingling, memory_problems,
                pale_skin, multiple_deficiencies]
    symptoms_count = sum(symptoms)

    row = {
        'age': age, 'bmi': bmi, 'symptoms_count': symptoms_count,
        'has_night_blindness': int(night_blindness),
        'has_fatigue': int(fatigue),
        'has_bleeding_gums': int(bleeding_gums),
        'has_bone_pain': int(bone_pain),
        'has_muscle_weakness': int(muscle_weakness),
        'has_numbness_tingling': int(numbness_tingling),
        'has_memory_problems': int(memory_problems),
        'has_pale_skin': int(pale_skin),
        'has_multiple_deficiencies': int(multiple_deficiencies),
    }

    categorical_values = {
        'gender': gender, 'smoking_status': smoking,
        'alcohol_consumption': alcohol_mapped, 'exercise_level': exercise,
        'diet_type': diet, 'sun_exposure': sun,
        'income_level': income, 'latitude_region': latitude,
    }

    # initialize all feature columns to 0
    for col in feature_columns:
        if col not in row:
            row[col] = 0

    # fixed: check against feature_columns, not row
    for cat_name, cat_value in categorical_values.items():
        col_name = f'{cat_name}_{cat_value}'
        if col_name in feature_columns:
            row[col_name] = 1

    df_input = pd.DataFrame([row])[feature_columns]
    df_input[continuous_cols] = scaler_obj.transform(df_input[continuous_cols])

    proba = model.predict_proba(df_input)[0]
    pred_idx = np.argmax(proba)
    pred_class = class_names[pred_idx]
    confidence = proba[pred_idx]

    conf_dict = {class_names[i]: float(proba[i]) for i in range(len(class_names))}
    rec = nutrient_info.get(pred_class, nutrient_info['Healthy'])

    diagnosis_md = f"""
## {pred_class.replace('_', ' ').lower()}

**confidence: {confidence*100:.1f}%**

---

**likely deficiency:** {rec['deficiency']}

**key foods:** {rec['foods']}

**quick tip:** {rec['tip']}
"""

    meal_plan = generate_meal_plan(pred_class, diet, allergies)

    return conf_dict, diagnosis_md, meal_plan


# custom css — dark green + black
css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

* { box-sizing: border-box; }

.gradio-container {
    max-width: 1100px !important;
    margin: auto !important;
    background: #000000 !important;
    font-family: 'Inter', sans-serif !important;
    color: #d1d5db !important;
}

#header-row {
    background: #000000;
    border-radius: 16px;
    padding: 40px 36px 32px 36px;
    margin-bottom: 20px;
    border: 1px solid #1a2e1f;
    border-top: 3px solid #22c55e;
}
#header-row h1 {
    color: #4ade80 !important;
    font-size: 2.4em !important;
    margin-bottom: 4px !important;
    letter-spacing: -0.5px;
}
#header-row p {
    color: #9ca3af !important;
    font-size: 15px !important;
    margin: 0 !important;
    line-height: 1.6 !important;
}
#header-row em {
    color: #4b5563 !important;
    font-size: 13px !important;
}

/* all inputs */
input, textarea, select,
.gr-input, .gr-dropdown,
.gr-slider input,
input[type="text"],
input[type="number"],
input[type="range"] {
    background: #0a0a0a !important;
    border: 1px solid #1c1c1c !important;
    color: #d1d5db !important;
    border-radius: 8px !important;
}
input:focus, textarea:focus, select:focus {
    border-color: #22c55e !important;
    box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.12) !important;
    outline: none !important;
}

/* primary button */
.gr-button-primary, button.primary {
    background: #16a34a !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 14px 28px !important;
    font-size: 15px !important;
    font-weight: 600 !important;
    color: #fff !important;
    transition: all 0.2s ease !important;
}
.gr-button-primary:hover, button.primary:hover {
    background: #22c55e !important;
    box-shadow: 0 4px 16px rgba(34, 197, 94, 0.25) !important;
}

/* section titles */
.section-title {
    color: #4ade80 !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 10px !important;
    padding-bottom: 8px;
    border-bottom: 1px solid #1a2e1f;
}

/* labels and text */
label, .gr-checkbox label, .gr-radio label {
    color: #9ca3af !important;
    font-size: 13px !important;
}

/* checkboxes — fixed */
input[type="checkbox"] {
    accent-color: #22c55e !important;
    width: 16px !important;
    height: 16px !important;
}

input[type="checkbox"]:checked {
    background-color: #22c55e !important;
    border-color: #22c55e !important;
}

input[type="checkbox"]:checked + label,
input[type="checkbox"]:checked ~ label,
.gr-checkbox:has(input:checked) label,
label:has(input[type="checkbox"]:checked) {
    color: #4ade80 !important;
}

/* radio buttons — fixed */
input[type="radio"] {
    accent-color: #22c55e !important;
}

/* ADD THIS */
input[type="radio"]:checked {
    accent-color: #22c55e !important;
    background-color: #22c55e !important;
    border-color: #22c55e !important;
    box-shadow: inset 0 0 0 4px #000000, 0 0 0 2px #22c55e !important;
}

input[type="radio"]:checked + label,
input[type="radio"]:checked ~ label,
.gr-radio:has(input:checked) label,
label:has(input[type="radio"]:checked) {
    color: #4ade80 !important;
}


/* label output component */
.gr-label {
    background: #0a0a0a !important;
    border: 1px solid #1c1c1c !important;
    border-radius: 12px !important;
}

/* markdown output */
.prose { color: #d1d5db !important; }
.prose h2 { color: #4ade80 !important; border-bottom: 1px solid #1a2e1f; padding-bottom: 8px; }
.prose h3 { color: #86efac !important; }
.prose strong { color: #e5e7eb !important; }
.prose hr { border-color: #1a2e1f !important; }
.prose li { color: #d1d5db !important; }
.prose p { color: #d1d5db !important; }
.prose a { color: #4ade80 !important; }

/* tabs */
.tabs .tab-nav { border-bottom: 1px solid #1c1c1c !important; }
.tabs .tab-nav button {
    color: #6b7280 !important;
    font-weight: 600 !important;
    background: transparent !important;
    border: none !important;
    padding: 10px 20px !important;
    transition: color 0.2s !important;
}
.tabs .tab-nav button:hover { color: #9ca3af !important; }
.tabs .tab-nav button.selected {
    color: #4ade80 !important;
    border-bottom: 2px solid #4ade80 !important;
}

/* disclaimer */
.disclaimer {
    background: #0a0f0a !important;
    border: 1px solid #1a2e1f !important;
    border-left: 3px solid #ca8a04 !important;
    border-radius: 8px !important;
    padding: 14px 18px !important;
    margin-top: 16px;
    color: #6b7280 !important;
    font-size: 13px !important;
}
.disclaimer strong { color: #ca8a04 !important; }

/* panels and backgrounds */
.block, .gr-panel, .gr-form, .gr-box, .gr-padded,
.panel, .form, .wrap, .container {
    background: #000000 !important;
    border-color: #1c1c1c !important;
}
.gr-box { border-radius: 12px !important; }

/* slider track */
input[type="range"]::-webkit-slider-runnable-track {
    background: #1a2e1f !important;
}
input[type="range"]::-webkit-slider-thumb {
    background: #22c55e !important;
}

/* dividers */
hr { border-color: #1c1c1c !important; }

/* scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0a0a0a; }
::-webkit-scrollbar-thumb { background: #1a2e1f; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #22c55e; }
"""

# build interface
with gr.Blocks(css=css, title="vitacheck", theme=gr.themes.Base()) as demo:

    with gr.Row(elem_id="header-row"):
        gr.Markdown("""
# vitacheck
**symptom-based vitamin deficiency detection & personalized meal planning**

virginia tech hci capstone — powered by xgboost + llama 3.3
        """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("<div class='section-title'>demographics</div>")
            age = gr.Slider(18, 84, value=30, step=1, label="age")
            gender = gr.Radio(["Male", "Female"], value="Male", label="gender")
            bmi = gr.Slider(15.0, 45.0, value=24.0, step=0.5, label="bmi")

            gr.Markdown("<div class='section-title'>lifestyle</div>")
            smoking = gr.Dropdown(["Never", "Former", "Current"], value="Never", label="smoking")
            alcohol = gr.Dropdown(["None", "Moderate", "Heavy", "Prefer not to say"],
                                  value="None", label="alcohol consumption")
            exercise = gr.Dropdown(["Sedentary", "Light", "Moderate", "Active"],
                                   value="Moderate", label="exercise")
            diet = gr.Dropdown(["Omnivore", "Vegetarian", "Vegan", "Pescatarian"],
                               value="Omnivore", label="diet type")

            gr.Markdown("<div class='section-title'>environment</div>")
            sun = gr.Dropdown(["Low", "Moderate", "High"], value="Moderate", label="sun exposure")
            income = gr.Dropdown(["Low", "Middle", "High"], value="Middle", label="income level")
            latitude = gr.Dropdown(["Low", "Mid", "High"], value="Mid", label="latitude region")

        with gr.Column(scale=1):
            gr.Markdown("<div class='section-title'>symptoms — check all that apply</div>")
            night_blindness = gr.Checkbox(label="night blindness / difficulty seeing in low light")
            fatigue = gr.Checkbox(label="fatigue / feeling unusually tired")
            bleeding_gums = gr.Checkbox(label="bleeding gums")
            bone_pain = gr.Checkbox(label="bone pain")
            muscle_weakness = gr.Checkbox(label="muscle weakness")
            numbness_tingling = gr.Checkbox(label="numbness or tingling")
            memory_problems = gr.Checkbox(label="memory problems / brain fog")
            pale_skin = gr.Checkbox(label="pale skin")
            multiple_deficiencies = gr.Checkbox(label="multiple symptoms / generally unwell")

            gr.Markdown("<div class='section-title'>dietary preferences</div>")
            allergies = gr.Textbox(
                label="allergies or restrictions",
                placeholder="e.g. lactose intolerant, nut allergy, gluten-free...",
                lines=2
            )

            submit_btn = gr.Button("analyze & generate meal plan", variant="primary", size="lg")

    gr.Markdown("---")

    with gr.Tabs():
        with gr.Tab("diagnosis"):
            with gr.Row():
                with gr.Column(scale=1):
                    confidence_output = gr.Label(label="confidence by class", num_top_classes=5)
                with gr.Column(scale=1):
                    diagnosis_output = gr.Markdown()

        with gr.Tab("personalized meal plan"):
            meal_plan_output = gr.Markdown()

    gr.Markdown("""
<div class='disclaimer'>
<strong>medical disclaimer:</strong> vitacheck is a screening tool for educational purposes only.
it is not a substitute for professional medical advice, diagnosis, or treatment.
always consult a qualified healthcare provider for proper evaluation.
</div>
    """)

    submit_btn.click(
        fn=predict,
        inputs=[age, gender, bmi, smoking, alcohol, exercise, diet, sun,
                income, latitude, night_blindness, fatigue, bleeding_gums,
                bone_pain, muscle_weakness, numbness_tingling, memory_problems,
                pale_skin, multiple_deficiencies, allergies],
        outputs=[confidence_output, diagnosis_output, meal_plan_output]
    )

demo.launch(share=True, debug=True)
