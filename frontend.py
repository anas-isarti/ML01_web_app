import requests
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from tensorflow import keras
import cv2
import numpy as np

st.title("MNIST Digit Recognizer")

SIZE = 192

canvas_result = st_canvas(
    fill_color="#ffffff",
    stroke_width=10,
    stroke_color='#ffffff',
    background_color="#000000",
    height=150,
    width=150,
    drawing_mode='freedraw',
    key="canvas",
)

if canvas_result.image_data is not None:
    img_color = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    img_rescaling = cv2.resize(img_color, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Input Image')
    st.image(img_rescaling)

if st.button('Predict'):
    url = 'http://localhost:8000/predict/'
    data_sent = cv2.imencode('.png', img_color)[1].tobytes()
    files = {'img': data_sent}
    response = requests.post(url, files=files)
    data_received = response.json()
    result = data_received['result']
    percent = data_received['percent']
    st.write(f'result: {result}')
    st.bar_chart(percent)