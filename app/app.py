import base64
import os
import uuid
import re
import torch
import numpy as np
from PIL import Image
import streamlit as st
from torch.utils import model_zoo
import torchvision.transforms as transforms

import sys
sys.path.append('.')
from src.models.dragan import Generator
from pretrained_superresolution.generator import srgan

st.set_option("deprecation.showfileUploaderEncoding", False)

@st.cache
def get_generator(url):
    model = Generator(hidden_dim=64)
    state_dict = model_zoo.load_url(url, progress=True, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

@st.cache
def get_super_resolution():
    model = srgan(pretrained=True)
    model.eval()
    return model

def generate_random_noise():
    return torch.randn((1, 100))

def download_image(object_to_download, filename, button_text):
    """
    Generates a link to download the given object_to_download.
    Params:
    ------
    object_to_download:  The object to be downloaded.
    filename (str): filename and extension of file. e.g. face1.jpg
    button_text (str): Text to display on download button (e.g. 'click here to download file')
    Returns:
    -------
    (str): the anchor tag to download object_to_download

    Examples:
    --------
    download_link(your_img, 'face1.jpg', 'Click to download image!')
    """
    try:
        # some strings <-> bytes conversions necessary here
        b64 = base64.b64encode(object_to_download.encode()).decode()

    except AttributeError as e:
        b64 = base64.b64encode(object_to_download).decode()

    button_uuid = str(uuid.uuid4()).replace('-', '')
    button_id = re.sub('\d+', '', button_uuid)

    custom_css = f""" 
        <style>
            #{button_id} {{
                background-color: rgb(255, 255, 255);
                color: rgb(38, 39, 48);
                padding: 0.25em 0.38em;
                position: relative;
                text-decoration: none;
                border-radius: 4px;
                border-width: 1px;
                border-style: solid;
                border-color: rgb(230, 234, 241);
                border-image: initial;

            }} 
            #{button_id}:hover {{
                border-color: rgb(246, 51, 102);
                color: rgb(246, 51, 102);
            }}
            #{button_id}:active {{
                box-shadow: none;
                background-color: rgb(246, 51, 102);
                color: white;
                }}
        </style> """

    dl_link = custom_css + f'<a download="{filename}" id="{button_id}" href="data:file/txt;base64,{b64}">{button_text}</a><br></br>'
    return dl_link


def main():
    CHECKPOINT_URL = "https://github.com/duylebkHCM/Anime-Face-Generator-/releases/download/v1.0/DRAGAN_G.pkl"
    generator = get_generator(CHECKPOINT_URL)
    generator = torch.jit.script(generator)
    # super_resolution_gen = get_super_resolution()
    # super_resolution_gen = torch.jit.script(super_resolution_gen)

    st.title('Anime Face Generator')
    st.header('PyTorch Project using Generative Adversarial Network')

    if st.button('Generate Some Random Faces'):
        noise = generate_random_noise()
        #Generate image
        with st.spinner('Wait for it...'):
            with torch.no_grad():
                image = generator(noise)
        image = image.data.numpy().transpose(0, 2, 3, 1)[0]
        image = (image + 1) / 2
        image = image*255.0
        image = image.astype('uint8')

        #@TODO 
        #Perform super_resolution_gen 
        # tensor = transforms.ToTensor()(image)
        # input_tensor = tensor.unsqueeze(0)
        # with torch.no_grad():
        #     sr_image = super_resolution_gen(input_tensor)

        # image = sr_image.data.numpy().transpose(0, 2, 3, 1)[0]
        # image = (image + 1) / 2
        # image = image*255.0
        # image = image.astype('uint8')
        pil_image = Image.fromarray(image).convert('RGB')
        #Save image 
        pil_image.save('generate_face.jpg', quality=100, subsampling=0)

        image = pil_image.resize((pil_image.size[0]*2, pil_image.size[1]*2), resample=Image.BICUBIC)
        image = np.asarray(image)
        st.image(image, use_column_width=True, caption='Generated Face')

        #Download the image    
        with open('generate_face.jpg', 'rb') as f:
            s = f.read()
        download_button_str = download_image(s, 'generate_face.jpg', f'Click here to download')
        st.markdown(download_button_str, unsafe_allow_html=True)
        os.remove('generate_face.jpg')


if __name__ == '__main__':
    main()
