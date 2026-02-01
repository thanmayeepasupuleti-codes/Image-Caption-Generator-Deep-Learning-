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
# Streamlit config
# -------------------------
st.set_page_config(page_title="Image Caption Generator")

# -------------------------
# Google Drive model config
# -------------------------
MODEL_PATH = "my_model.keras"
MODEL_URL = "https://drive.google.com/uc?id=13J2Aujk-1oqVRsQ_Vtpq_3AIMIdqbFtk"

# -------------------------
# Load tokenizer
# -------------------------
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

index_word = {v: k for k, v in tokenizer.word_index.items()}
max_caption_length = 34  # same as training

# -------------------------
# Custom Attention Layer
# -------------------------
class BahdanauAttention(Layer):
    def __init__(self, units, **kwargs):
        super(BahdanauAttention, self).__init__(**kwargs)
        self.units = units
        self.W1 = Dense(units)
        self.W2 = Dense(units)
        self.V = Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# -------------------------
# Load caption model (DOWNLOAD + CACHE)
# -------------------------
@st.cache_resource
def load_caption_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading caption model..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model = tf.keras.models.load_model(
        MODEL_PATH,
        custom_objects={"BahdanauAttention": BahdanauAttention},
        compile=False,
        safe_mode=False
    )
    return model

caption_model = load_caption_model()

# -------------------------
# Load VGG16 (CACHE)
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
# Beam Search Captioning
# -------------------------
def predict_caption_beam_search(model, image_feature, tokenizer, max_len, beam_width=3):
    start = tokenizer.word_index["startseq"]
    end = tokenizer.word_index["endseq"]

    image_feature = image_feature.reshape((1, 196, 512))
    sequences = [[[start], 0.0]]

    for _ in range(max_len):
        all_candidates = []

        for seq, score in sequences:
            if seq[-1] == end:
                all_candidates.append((seq, score))
                continue

            padded = pad_sequences([seq], maxlen=max_len)
            preds = model.predict([image_feature, padded], verbose=0)[0]

            top_words = np.argsort(preds)[-beam_width:]

            for word in top_words:
                candidate = seq + [word]
                candidate_score = score - np.log(preds[word] + 1e-10)
                all_candidates.append((candidate, candidate_score))

        sequences = sorted(all_candidates, key=lambda x: x[1])[:beam_width]

    final_seq = sequences[0][0]

    caption = []
    for idx in final_seq:
        word = index_word.get(idx)
        if word == "endseq":
            break
        caption.append(word)

    return " ".join(caption)

# -------------------------
# Generate caption
# -------------------------
def generate_caption(image, beam_width=3):
    image = image.resize((224, 224))
    image = np.array(image)

    if image.shape[-1] == 4:
        image = image[:, :, :3]

    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)

    feature_map = vgg.predict(image, verbose=0)
    feature_map = feature_map.reshape((196, 512))

    caption = predict_caption_beam_search(
        caption_model,
        feature_map,
        tokenizer,
        max_caption_length,
        beam_width
    )

    caption = caption.replace("startseq", "").replace("endseq", "")
    return caption.strip()

# -------------------------
# Streamlit UI
# -------------------------
st.title("🖼️ Image Caption Generator (Beam Search)")
st.write("Upload an image and generate a caption using Beam Search")

beam_width = st.slider("Beam Width", 2, 5, 3)

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Generate Caption"):
        with st.spinner("Generating caption..."):
            caption = generate_caption(image, beam_width)
        st.success(caption)
