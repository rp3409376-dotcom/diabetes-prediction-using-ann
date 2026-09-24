# # Copyright 2015 The TensorFlow Authors. All Rights Reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
# # ==============================================================================

# """Generic entry point script."""
# import sys as _sys

# from absl.app import run as _run

# from tensorflow.python.platform import flags
# from tensorflow.python.util.tf_export import tf_export


# def _parse_flags_tolerate_undef(argv):
#   """Parse args, returning any unknown flags (ABSL defaults to crashing)."""
#   return flags.FLAGS(_sys.argv if argv is None else argv, known_only=True)


# @tf_export(v1=['app.run'])
# def run(main=None, argv=None):
#   """Runs the program with an optional 'main' function and 'argv' list."""

#   main = main or _sys.modules['__main__'].main

#   _run(main=main, argv=argv, flags_parser=_parse_flags_tolerate_undef)





import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon="🩺",
    layout="wide"
)

# ==========================================
# LOAD MODEL
# ==========================================
@st.cache_resource
def load_diabetes_model():
    return load_model("diabetes_model.keras")


try:
    model = load_diabetes_model()
except Exception as e:
    st.error("❌ Model load nahi ho raha.")
    st.code(str(e))
    st.stop()


# ==========================================
# TITLE
# ==========================================
st.title("🩺 Diabetes Prediction System")
st.write("Enter the patient's information below to predict diabetes.")
st.divider()


# ==========================================
# INPUT FEATURES
# ==========================================
col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        max_value=300,
        value=120,
        step=1
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        max_value=200,
        value=70,
        step=1
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        max_value=100,
        value=20,
        step=1
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0,
        max_value=900,
        value=80,
        step=1
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.47,
        step=0.01
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )


# ==========================================
# PREDICT BUTTON
# ==========================================
st.divider()

if st.button("🔍 Predict Diabetes", use_container_width=True):

    # Input data
    input_data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age
    ]], dtype=np.float32)

    try:

        # Prediction
        prediction = model.predict(input_data, verbose=0)

        probability = float(prediction[0][0])

        st.subheader("Prediction Result")

        # Result
        if probability >= 0.5:

            st.error(
                "⚠️ The model predicts that the person may have diabetes."
            )

        else:

            st.success(
                "✅ The model predicts that the person may not have diabetes."
            )

        # Probability
        st.metric(
            "Prediction Probability",
            f"{probability * 100:.2f}%"
        )

    except Exception as e:

        st.error("❌ Prediction failed.")
        st.code(str(e))