import argparse
import os
import sys
import pickle
import math

import numpy as np



if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Tensorflow to pytorch model checkpoint converter"
    )
    parser.add_argument(
        "--gen", action="store_true", help="convert the generator weights"
    )
    parser.add_argument(
        "--disc", action="store_true", help="convert the discriminator weights"
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier factor. config-f = 2, else = 1",
    )
    parser.add_argument("path", metavar="PATH", help="path to the tensorflow weights")

    args = parser.parse_args()

    import torch
    from torchvision import utils
    from model import Generator, Discriminator

    name = os.path.splitext(os.path.basename(args.path))[0]
    with open(name+"_numpy.pkl", "rb") as f:
        state_dict = pickle.load( f)
    size = state_dict.pop('size')
    n_mlp = state_dict.pop('n_mlp')
    z = state_dict.pop('z')
    n_sample = state_dict.pop('n_sample')
    latent_avg = state_dict.pop('latent_avg')
    #import pdb
    #pdb.set_trace()

    g = Generator(size, 512, n_mlp, channel_multiplier=args.channel_multiplier)
    for k,v  in state_dict.items():
        state_dict[k] = torch.from_numpy(v)
    g.load_state_dict(state_dict, strict=False)

    latent_avg = torch.from_numpy(latent_avg)

    ckpt = {"g_ema": state_dict, "latent_avg": latent_avg}

    if args.gen:
        raise NotImplementedError
        g_train = Generator(size, 512, n_mlp, channel_multiplier=args.channel_multiplier)
        g_train_state = g_train.state_dict()
        g_train_state = fill_statedict(g_train_state, generator.vars, size, n_mlp)
        ckpt["g"] = g_train_state

    if args.disc:
        raise NotImplementedError
        disc = Discriminator(size, channel_multiplier=args.channel_multiplier)
        d_state = disc.state_dict()
        d_state = discriminator_fill_statedict(d_state, discriminator.vars, size)
        ckpt["d"] = d_state

    name = os.path.splitext(os.path.basename(args.path))[0]
    torch.save(ckpt, name + ".pt")


    g = g.to(device)

    z = np.random.RandomState(0).randn(n_sample, 512).astype("float32")

    with torch.no_grad():
        img_pt, _ = g(
            [torch.from_numpy(z).to(device)],
            truncation=0.5,
            truncation_latent=latent_avg.to(device),
            randomize_noise=False,
        )

    #Gs_kwargs = dnnlib.EasyDict()
    #Gs_kwargs.randomize_noise = False
    #img_tf = g_ema.run(z, None, **Gs_kwargs)
    #img_tf = torch.from_numpy(img_tf).to(device)

    #img_diff = ((img_pt + 1) / 2).clamp(0.0, 1.0) - ((img_tf.to(device) + 1) / 2).clamp(
    #    0.0, 1.0
    #)

    #img_concat = torch.cat((img_tf, img_pt, img_diff), dim=0)

    #print(img_diff.abs().max())

    utils.save_image(
        img_pt, name + ".png", nrow=n_sample, normalize=True, range=(-1, 1)
    )
