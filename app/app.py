import streamlit as st
import torch
from torch.utils import model_zoo
from PIL import Image
import numpy as np
import sys
sys.path.append('.')
from src.models.dragan import Generator

st.set_option("deprecation.showfileUploaderEncoding", False)

CHECKPOINT_URL = ""

@st.cache
def get_generator():
    model = Generator(hidden_dim=64)
    state_dict = model_zoo.load_url(CHECKPOINT_URL, progress=True, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    script_module = torch.jit.script(model)
    return script_module


generator = get_generator()

st.title('Anime Face Generator')

def generate_random_noise():
    return torch.randn((1, 100))

if st.button('Generate Random Noise'):
    noise = generate_random_noise()
    #Generate image
    st.write('Generating face...')
    with torch.no_grad():
        image = generator(noise)
    image = image.data.numpy().transpose(0, 2, 3, 1)[0]
    image = (image + 1) / 2
    image = image.astype('uint8')
    image = Image.fromarray(image).convert('RGB')
    image = image.resize((image.size[0]*2, image.size[1]*2), resample=Image.BILINEAR)
    image = np.asarray(image)
    st.image(image, use_column_width=True, caption='Generated Face')
