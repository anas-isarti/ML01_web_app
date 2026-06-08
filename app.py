import streamlit as st
from streamlit_drawable_canvas import st_canvas
import cv2
import numpy as np

st.set_page_config(page_title="MNIST Digit Recognizer", page_icon="✏️")
st.title("MNIST Digit Recognizer")
st.markdown("Draw a digit (0-9) on the canvas below and let the model predict it!")

if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = 0

@st.cache_resource
def load_weights():
    data = np.load('mnist_weights.npz')
    return data['w1'], data['b1'], data['w2'], data['b2']

w1, b1, w2, b2 = load_weights()

def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def predict(img_flat):
    hidden = relu(np.dot(img_flat, w1) + b1)
    output = softmax(np.dot(hidden, w2) + b2)
    return output

col_canvas, col_clear = st.columns([3, 1])
with col_canvas:
    canvas_result = st_canvas(
        fill_color="#ffffff",
        stroke_width=10,
        stroke_color='#ffffff',
        background_color="#000000",
        height=150,
        width=150,
        drawing_mode='freedraw',
        key=f"canvas_{st.session_state.canvas_key}",
    )
with col_clear:
    st.write("")
    st.write("")
    if st.button("Clear"):
        st.session_state.canvas_key += 1
        st.rerun()

if canvas_result.image_data is not None:
    img_data = canvas_result.image_data
    if img_data.dtype == np.float32 or img_data.dtype == np.float64:
        img_data = (img_data * 255).astype(np.uint8)
    else:
        img_data = img_data.astype(np.uint8)
    img = cv2.resize(img_data, (28, 28))
    col1, col2 = st.columns(2)
    with col1:
        st.write("Input Image")
        img_display = cv2.resize(img, (192, 192), interpolation=cv2.INTER_NEAREST)
        st.image(img_display, clamp=True)

    if st.button('Predict'):
        img_grey = img[:, :, 0].astype(np.float32)
        img_flat = img_grey.reshape(1, 784)
        pred = predict(img_flat)
        result = int(np.argmax(pred[0]))
        confidence = float(np.max(pred[0]))
        with col2:
            st.write(f"## Prediction: **{result}**")
            st.write(f"Confidence: **{confidence:.2%}**")
            st.bar_chart(pred[0], height=200)