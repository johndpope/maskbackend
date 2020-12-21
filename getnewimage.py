import subprocess as sp
import dnnlib
from run_generator import generate_images,generate_average_images,generate_blended_image
import random

# def generate_images(network_pkl, seeds, npy_files, truncation_psi):
dnnlib.tflib.init_tf()

def getnewImage():

    # img = generate_average_images('AlfredENeuman24_ADA-VersatileFaces36_ADA_v2-blended-64.pkl')
    # img = generate_images('AlfredENeuman24_ADA-VersatileFaces36_ADA_v2-blended-64.pkl','0_bg.png')
    img = generate_blended_image()
    return img