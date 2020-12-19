import subprocess as sp
import dnnlib
from run_generator import generate_images
import random

# def generate_images(network_pkl, seeds, npy_files, truncation_psi):
dnnlib.tflib.init_tf()

def getnewImage():

    img = generate_images('FFHQ-CartoonsAlignedHQ36v2.pkl')
   
    return img