import streamlit as st
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Layer, Dense
import gdown
import os

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Image Caption Generator",
    layout="centered"
)

# -------------------------
# Custom CSS (POLISH)
# -------------------------
st.markdown(
    """
    <style>
        body {
            background-color: #fafafa;
        }
        .main-header {
            background: linear-gradient(90deg, #3f51b5, #5c6bc0);
            padding: 30px 20px;
            border-radius: 14px;
            text-align: center;
            color: white;
            margin-bottom: 30px;
        }
        .main-header h1 {
            font-size: 2.4rem;
            margin-bottom: 8px;
        }
        .main-header p {
            font-size: 1.05rem;
            opacity: 0.9;
        }
        .card {
            background-color: white;
            padding: 20px;
            border-radius: 14px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.06);
            margin-bottom: 24px;
        }
        .caption-box {
            background-color: #f4f6fb;
            padding: 18px;
            border-radius: 12px;
            font-size: 1.05rem;
            text-align: center;
            border-left: 5px solid #3f51b5;
        }
        .footer {
            text-align: center;
            color: #6b6b6b;
            font-size: 0.9rem;
            margin-top: 40px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Header
# -------------------------
st.markdown(
    """
    <div class="main-header">
        <h1>Image Caption Generator</h1>
        <p>Deep Learning with Bahdanau Attention and Beam Search</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# Google Drive Model Config
# -------------------------
MODEL_PATH = "my_model.keras"
MODEL_URL = "https://drive.google.com/uc?id=13J2Aujk-1oqVRsQ_Vtpq_3AIMIdqbFtk"

# -------------------------
# Load Tokenizer
# -------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

index_word = {v: k for k, v in tokenizer.word_index.items()}
max_caption_length = 34

# -------------------------
# Custom Attention Layer
# -------------------------
class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden))
        weights = tf.nn.softmax(self.V(score), axis=1)
        context = tf.reduce_sum(weights * features, axis=1)
        return context, weights

# -------------------------
# Load Caption Model
# -------------------------
@st.cache_resource
def load_caption_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    return tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"BahdanauAttention": BahdanauAttention},
        compile=False,
        safe_mode=False
    )

caption_model = load_caption_model()

# -------------------------
# Load VGG16
# -------------------------
@st.cache_resource
def load_vgg():
    base = VGG16(weights="imagenet")
    return Model(
        inputs=base.inputs,
        outputs=base.get_layer("block5_conv3").output
    )

vgg = load_vgg()

# -------------------------
# Beam Search
# -------------------------
def predict_caption_beam_search(model, image_feature, tokenizer, max_len, beam_width):
    start = tokenizer.word_index["startseq"]
    end = tokenizer.word_index["endseq"]

    image_feature = image_feature.reshape((1, 196, 512))
    sequences = [[[start], 0.0]]

    for _ in range(max_len):
        candidates = []
        for seq, score in sequences:
            if seq[-1] == end:
                candidates.append((seq, score))
                continue

            padded = pad_sequences([seq], maxlen=max_len)
            preds = model.predict([image_feature, padded], verbose=0)[0]
            top = np.argsort(preds)[-beam_width:]

            for word in top:
                candidates.append(
                    (seq + [word], score - np.log(preds[word] + 1e-10))
                )

        sequences = sorted(candidates, key=lambda x: x[1])[:beam_width]

    final = sequences[0][0]
    caption = []

    for idx in final:
        word = index_word.get(idx)
        if word == "endseq":
            break
        caption.append(word)

    return " ".join(caption)

# -------------------------
# Generate Caption
# -------------------------
def generate_caption(image, beam_width):
    image = image.resize((224, 224))
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = preprocess_input(np.expand_dims(image, axis=0))
    features = vgg.predict(image, verbose=0).reshape((196, 512))

    caption = predict_caption_beam_search(
        caption_model, features, tokenizer, max_caption_length, beam_width
    )

    return caption.replace("startseq", "").replace("endseq", "").strip()

# -------------------------
# Controls Card
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Settings")
beam_width = st.slider("Beam Width", 2, 5, 3)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Upload Card
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Upload Image")
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, use_column_width=True)

    if st.button("Generate Caption", use_container_width=True):
        with st.spinner("Generating caption..."):
            caption = generate_caption(image, beam_width)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="caption-box">', unsafe_allow_html=True)
        st.write(caption)
        st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Team Section
# -------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Project Team")
st.markdown(
    """
    Thanmayee P — HU22CSEN0500235  
    Praharshitha K — HU22CSEN0500241  
    Anvitha N — HU22CSEN0500214  
    Phani — HU22CSEN0500156  
    """
)
st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown(
    """
    <div class="footer">
        Image Caption Generator using Deep Learning<br>
        Built with Streamlit and TensorFlow
    </div>
    """,
    unsafe_allow_html=True
)
