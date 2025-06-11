import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

st.markdown(
    """
    <div style="background-color:red; padding:10px">
        <h2 style="color:white; text-align:center;">X-Ray Image&nbsp;Classifier</h2>
    </div>
    """,
    unsafe_allow_html=True,
)

IMG_SIZE   = 100
CATEGORIES = ["NORMAL", "PNEUMONIA"]

@st.cache_resource(show_spinner=True)
def load_model():
    return tf.keras.models.load_model("Xray_pretrained_model.h5")

model = load_model()

def preprocess(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB")                 
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0)      

st.subheader("Upload an X-ray image (jpg / jpeg / png)")

file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if file is not None:
    pil_img = Image.open(BytesIO(file.read()))
    st.image(pil_img, caption="Input image", use_column_width=True)

    if st.button("Predict"):
        tensor = preprocess(pil_img)
        prob   = model.predict(tensor)[0][0]    
        label  = CATEGORIES[int(round(prob))]

        st.success(f"**{label}**  –  confidence {prob:.3f}")
