import subprocess as sp
import dnnlib
from run_generator import generate_images
import random

# def generate_images(network_pkl, seeds, npy_files, truncation_psi):
dnnlib.tflib.init_tf()

def getnewImage():
    seed = random.randint(1,1000)
    img = generate_images('FFHQ-CartoonsAlignedHQ36v2.pkl', [seed], None, 0.6)
   
    return img