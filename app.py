import gradio as gr
import numpy as np
from utils.preprocessing import load_data, scale_data
from models.dnn import build_dnn

# =========================
# LOAD DATA + MODEL
# =========================
X_train, X_test, y_train, y_test = load_data()
X_train, X_test = scale_data(X_train, X_test)

model = build_dnn(X_train.shape[1])
model.fit(X_train, y_train, epochs=10, verbose=0)

mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
std[std == 0] = 1


# =========================
# PREDICTION FUNCTION
# =========================
def predict(age, smoking, yellow_fingers, anxiety, peer_pressure,
            chronic_disease, fatigue, allergy, wheezing,
            alcohol, coughing, breath, swallowing, chest_pain):

    inputs = [
        age, smoking, yellow_fingers, anxiety, peer_pressure,
        chronic_disease, fatigue, allergy, wheezing,
        alcohol, coughing, breath, swallowing, chest_pain
    ]

    # ✅ Basic sanity check
    if sum(inputs) == 0:
        return "Low Risk (No symptoms detected)"

    # =========================
    # PREPARE INPUT
    # =========================
    full_data = np.zeros(X_train.shape[1])

    for i in range(len(inputs)):
        full_data[i] = inputs[i]

    for i in range(len(inputs), X_train.shape[1]):
        full_data[i] = mean[i]

    full_data = full_data.reshape(1, -1)
    full_data = (full_data - mean) / std

    # =========================
    # MODEL PREDICTION
    # =========================
    prob = model.predict(full_data, verbose=0)[0][0]

    # =========================
    # SMOOTHING (avoid 0/1 extremes)
    # =========================
    prob = 0.2 + 0.6 * prob

    # =========================
    # SYMPTOM-BASED CALIBRATION
    # =========================
    symptom_count = sum(inputs[1:])  # exclude age

    if symptom_count <= 3:
        prob *= 0.6
    elif symptom_count <= 6:
        prob *= 0.8

    # =========================
    # FINAL DECISION
    # =========================
    if prob > 0.7:
        return f"High Risk ({prob:.2f})"
    elif prob > 0.5:
        return f"Moderate Risk ({prob:.2f})"
    else:
        return f"Low Risk ({prob:.2f})"


# =========================
# UI (MINIMAL PROFESSIONAL)
# =========================
with gr.Blocks(
    title="OncoAI – Lung Cancer Risk Assessment",
    theme=gr.themes.Base(),
    css="""
    .gradio-container {
        max-width: 900px !important;
        margin: auto;
        font-family: 'Inter', sans-serif;
    }
    .section {
        margin-top: 25px;
        margin-bottom: 10px;
        font-size: 13px;
        color: #666;
    }
    """
) as app:

    gr.Markdown("# Lung Cancer Risk Assessment")
    gr.Markdown("AI-powered clinical decision support system")

    with gr.Row():
        age = gr.Slider(10, 90, value=30, label="Age")

    gr.Markdown('<div class="section">Lifestyle</div>')
    with gr.Row():
        smoking = gr.Dropdown([0, 1], label="Smoking", value=0)
        alcohol = gr.Dropdown([0, 1], label="Alcohol Consumption", value=0)

    gr.Markdown('<div class="section">Medical & Behavioral</div>')
    with gr.Row():
        anxiety = gr.Dropdown([0, 1], label="Anxiety", value=0)
        peer_pressure = gr.Dropdown([0, 1], label="Peer Pressure", value=0)
        chronic_disease = gr.Dropdown([0, 1], label="Chronic Disease", value=0)

    gr.Markdown('<div class="section">Symptoms</div>')
    with gr.Row():
        yellow_fingers = gr.Dropdown([0, 1], label="Yellow Fingers", value=0)
        fatigue = gr.Dropdown([0, 1], label="Fatigue", value=0)
        allergy = gr.Dropdown([0, 1], label="Allergy", value=0)
        wheezing = gr.Dropdown([0, 1], label="Wheezing", value=0)

    with gr.Row():
        coughing = gr.Dropdown([0, 1], label="Coughing", value=0)
        breath = gr.Dropdown([0, 1], label="Shortness of Breath", value=0)
        swallowing = gr.Dropdown([0, 1], label="Swallowing Difficulty", value=0)
        chest_pain = gr.Dropdown([0, 1], label="Chest Pain", value=0)

    predict_btn = gr.Button("Assess Risk", variant="primary")

    output = gr.Textbox(label="Result", interactive=False)

    predict_btn.click(
        predict,
        inputs=[
            age, smoking, yellow_fingers, anxiety, peer_pressure,
            chronic_disease, fatigue, allergy, wheezing,
            alcohol, coughing, breath, swallowing, chest_pain
        ],
        outputs=output
    )

    gr.Markdown("""
    ---
    This tool is for research purposes only and does not replace medical diagnosis.
    """)

# =========================
# RUN
# =========================
app.launch()