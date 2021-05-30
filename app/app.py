import streamlit as st
import torch
import numpy as np
from PIL import Image
from pathlib import Path
# import sys
# sys.path.append('..')
from src.models.dcgan import Generator

logo_img = Path('../dataset/raw/faces/4.jpg')
img = Image.open(logo_img)

st.set_option("deprecation.showfileUploaderEncoding", False)

CHECKPOINT_PATH = Path('../checkpoints/DCGAN/DCGAN_G.pkl')

@st.cache
def get_generator():
    model = Generator(hidden_dim=64)
    model.load_state_dict(torch.load(str(CHECKPOINT_PATH)))
    model.eval()
    return model


generator = get_generator()

st.title('Anime Face Generator')

def generate_random_noise():
    return torch.randn((1, 100))

if st.button('Generate Random Noise'):
    noise = generate_random_noise()
    #Generate image
    with torch.no_grad():
        image = generator(noise)
    image = image.data.numpy().transpose(0, 2, 3, 1)[0]
    image = (image + 1) / 2

    st.image(image, use_column_width=True, caption='Generated Face')
