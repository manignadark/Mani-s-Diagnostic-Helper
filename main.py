import streamlit as st
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from PIL import Image

# ✅ Configure API key from secrets
genai.configure(api_key=st.secrets["general"]["api_key"])

# System Prompt
system_prompt = """
You are an expert medical image analysis AI...
(keep your full prompt here)
"""

# Model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
}

# Safety settings
safety_settings = [
    {"category": HarmCategory.HARM_CATEGORY_HATE_SPEECH, "threshold": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE},
    {"category": HarmCategory.HARM_CATEGORY_HARASSMENT, "threshold": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE},
    {"category": HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, "threshold": HarmBlockThreshold.BLOCK_NONE},
]

# ✅ Use the new Gemini model (flash = faster/cheaper, pro = more accurate)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",  # or "gemini-1.5-pro"
    generation_config=generation_config,
    safety_settings=safety_settings
)

# Streamlit Layout
st.set_page_config(page_title="Diagnostic Analyzer", page_icon=":robot:")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.image("maniganda.png", width=200)
    st.image("image.png", width=200)

st.title("Medical Image Diagnostic Analyzer")
st.write("Upload a medical image and I will analyze it for potential abnormalities.")

uploaded_file = st.file_uploader("Upload Medical Image", type=["png", "jpg", "jpeg", "bmp", "tiff"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Medical Image", use_container_width=True)

# ✅ Button always visible
if st.button("Analyze Image"):
    if uploaded_file is None:
        st.warning("⚠️ Please upload an image before analysis.")
    else:
        with st.spinner("Analyzing image..."):
            try:
                image = Image.open(uploaded_file)

                # ✅ Directly pass system prompt + image
                response = model.generate_content([system_prompt, image])

                # ✅ Safely get text
                try:
                    analysis_text = response.text
                except:
                    analysis_text = response.candidates[0].content.parts[0].text

                st.write("### 🩺 Analysis Report")
                st.write(analysis_text)

                st.info("⚠️ This analysis is for informational purposes only and is not a substitute for professional medical advice. Always consult a healthcare professional.")

            except Exception as e:
                st.error(f"An error occurred: {e}")
