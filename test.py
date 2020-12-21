import numpy as np
from PIL import Image
import dnnlib
import dnnlib.tflib as tflib
from pathlib import Path
from imgcat import imgcat
import pretrained_networks
import PIL.Image
latent_dir = Path("generated")
latents = latent_dir.glob("*.npy")


# blended_url = 'ffhq-cartoon-blended.pkl' # 
blended_url ="AlfredENeuman24_ADA-VersatileFaces36_ADA_v2-blended-64.pkl"
ffhq_url = "stylegan2-ffhq-config-f.pkl"

_, _, Gs_blended = pretrained_networks.load_networks(blended_url)
_, _, Gs = pretrained_networks.load_networks(ffhq_url)


for latent_file in latents:
  print("latent_file:",latent_file)
  latent = np.load(latent_file)
  latent = np.expand_dims(latent,axis=0)
  synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
  images = Gs_blended.components.synthesis.run(latent, randomize_noise=False, **synthesis_kwargs)
  file_name = latent_file.parent / (f"{latent_file.stem}-toon.jpg")
  Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').save(file_name)
  img = PIL.Image.open(file_name)
  imgcat(img)