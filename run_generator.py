
import pretrained_networks
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""

import argparse
import os
import pickle
import re

import numpy as np
import PIL.Image
import random
import dnnlib
import dnnlib.tflib as tflib
from projector import Projector
from imgcat import imgcat
import numpy as np
from pathlib import Path
from align_images import download_extract
#----------------------------------------------------------------------------

def generate_blended_image(image):
    tflib.init_tf()
    # download_extract()


    # SAVE FILE TO RAW

    # ALIGN IT


    print('Loading networks from "%s"...' % image)
    # _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    blended_url = "AlfredENeuman24_ADA-VersatileFaces36_ADA_v2-blended-64.pkl"
    ffhq_url = "stylegan2-ffhq-config-f.pkl"

    _, _, Gs_blended = pretrained_networks.load_networks(blended_url)
    _, _, Gs = pretrained_networks.load_networks(ffhq_url)

    # latent_dir = Path("generated")
    # latents = latent_dir.glob("*.npy")
    # for latent_file in latents:
    #     latent = np.load(latent_file)
    #     latent = np.expand_dims(latent,axis=0)
    #     synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False), minibatch_size=8)
    #     images = Gs_blended.components.synthesis.run(latent, randomize_noise=False, **synthesis_kwargs)
    #     Image.fromarray(images.transpose((0,2,3,1))[0], 'RGB').save(latent_file.parent / (f"{latent_file.stem}-toon.jpg"))
    # imgcat(img)
    
    
    return img


def generate_average_images(network_pkl):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)


    # os.makedirs(outdir, exist_ok=True)

    # GENERATE AVERAGE IMAGE
    dlatents_var = Gs.get_var('dlatent_avg')
    dlatents = np.tile(dlatents_var, [1, Gs.components.synthesis.input_shape[1], 1])

    Gs_syn_kwargs = dnnlib.EasyDict()
    Gs_syn_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8,
                                        nchw_to_nhwc=True)
    Gs_syn_kwargs.randomize_noise = False    
    image = Gs.components.synthesis.run(dlatents, **Gs_syn_kwargs)[0]
    img = PIL.Image.fromarray(image, 'RGB')
    imgcat(img)
    return img

def generate_images(network_pkl,target_fname):
    tflib.init_tf()
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)

    # # Render images for a given dlatent vector.
    if target_fname is not None:
            target_pil = PIL.Image.open(target_fname)
            w, h = target_pil.size
            s = min(w, h)
            target_pil = target_pil.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
            target_pil= target_pil.convert('RGB')
            target_pil = target_pil.resize((Gs.output_shape[3], Gs.output_shape[2]), PIL.Image.ANTIALIAS)
            target_uint8 = np.array(target_pil, dtype=np.uint8)
            target_float = target_uint8.astype(np.float32).transpose([2, 0, 1]) * (2 / 255) - 1

            proj = Projector()
            proj.set_network(Gs)
            proj.start([target_float])
            proj.dlatents
            
            dlatents = proj.dlatents #np.load(dlatents_npz)['dlatents']
            assert dlatents.shape[1:] == (18, 512) # [N, 18, 512]
            imgs = Gs.components.synthesis.run(dlatents, output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True))
            for i, img in enumerate(imgs):
                #fname = f'{outdir}/dlatent{i:02d}.png'
                #print (f'Saved {fname}')
                return PIL.Image.fromarray(img, 'RGB')#.save(fname)


    # Render images for dlatents initialized from random seeds.
    Gs_kwargs = {
        'output_transform': dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True),
        'randomize_noise': False
    }
    # if truncation_psi is not None:
    #     Gs_kwargs['truncation_psi'] = truncation_psi

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    label = np.zeros([1] + Gs.input_shapes[1][1:])
    # if class_idx is not None:
    #     label[:, class_idx] = 1

    # for seed_idx, seed in enumerate(seeds):
        # print('Generating image for seed %d (%d/%d) ...' % (seed, seed_idx, len(seeds)))
    seed = np.random.randint(0, 1000)
    rnd = np.random.RandomState(seed)
    z = rnd.randn(1, *Gs.input_shape[1:]) # [minibatch, component]
    tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
    images = Gs.run(z, label, **Gs_kwargs) # [minibatch, height, width, channel]
    return PIL.Image.fromarray(images[0], 'RGB')#.save(f'{outdir}/seed{seed:04d}.png')

#----------------------------------------------------------------------------

def _parse_num_range(s):
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

_examples = '''examples:

  # Generate curated MetFaces images without truncation (Fig.10 left)
  python %(prog)s --outdir=out --trunc=1 --seeds=85,265,297,849 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
  python %(prog)s --outdir=out --trunc=0.7 --seeds=600-605 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/metfaces.pkl

  # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
  python %(prog)s --outdir=out --trunc=1 --seeds=0-35 --class=1 \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/cifar10.pkl

  # Render image from projected latent vector
  python %(prog)s --outdir=out --dlatents=out/dlatents.npz \\
      --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada/pretrained/ffhq.pkl
'''

#----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate images using pretrained network pickle.',
        epilog=_examples,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--seeds', type=_parse_num_range, help='List of random seeds')
    g.add_argument('--dlatents', dest='dlatents_npz', help='Generate images for saved dlatents')
    parser.add_argument('--trunc', dest='truncation_psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser.add_argument('--class', dest='class_idx', type=int, help='Class label (default: unconditional)')
    parser.add_argument('--outdir', help='Where to save the output images', required=True, metavar='DIR')

    args = parser.parse_args()
    generate_images(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
