import argparse
import os
import sys
import pickle
import math

import numpy as np


def convert_modconv(vars, source_name, target_name, flip=False):
    weight = vars[source_name + "/weight"].value().eval()
    mod_weight = vars[source_name + "/mod_weight"].value().eval()
    mod_bias = vars[source_name + "/mod_bias"].value().eval()
    noise = vars[source_name + "/noise_strength"].value().eval()
    bias = vars[source_name + "/bias"].value().eval()

    dic = {
        "conv.weight": np.expand_dims(weight.transpose((3, 2, 0, 1)), 0),
        "conv.modulation.weight": mod_weight.transpose((1, 0)),
        "conv.modulation.bias": mod_bias + 1,
        "noise.weight": np.array([noise]),
        "activate.bias": bias,
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + "." + k] = v#torch.from_numpy(v)

    if flip:
        dic_torch[target_name + ".conv.weight"] = \
        np.flip(
            dic_torch[target_name + ".conv.weight"], [3, 4]
        )
        #torch.flip(
        #    dic_torch[target_name + ".conv.weight"], [3, 4]
        #)

    return dic_torch


def convert_conv(vars, source_name, target_name, bias=True, start=0):
    weight = vars[source_name + "/weight"].value().eval()

    dic = {"weight": weight.transpose((3, 2, 0, 1))}

    if bias:
        dic["bias"] = vars[source_name + "/bias"].value().eval()

    dic_torch = {}

    dic_torch[target_name + f".{start}.weight"] = dic["weight"]#torch.from_numpy(dic["weight"])

    if bias:
        dic_torch[target_name + f".{start + 1}.bias"] = dic["bias"]#torch.from_numpy(dic["bias"])

    return dic_torch


def convert_torgb(vars, source_name, target_name):
    weight = vars[source_name + "/weight"].value().eval()
    mod_weight = vars[source_name + "/mod_weight"].value().eval()
    mod_bias = vars[source_name + "/mod_bias"].value().eval()
    bias = vars[source_name + "/bias"].value().eval()

    dic = {
        "conv.weight": np.expand_dims(weight.transpose((3, 2, 0, 1)), 0),
        "conv.modulation.weight": mod_weight.transpose((1, 0)),
        "conv.modulation.bias": mod_bias + 1,
        "bias": bias.reshape((1, 3, 1, 1)),
    }

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + "." + k] = v#torch.from_numpy(v)

    return dic_torch


def convert_dense(vars, source_name, target_name):
    weight = vars[source_name + "/weight"].value().eval()
    bias = vars[source_name + "/bias"].value().eval()

    dic = {"weight": weight.transpose((1, 0)), "bias": bias}

    dic_torch = {}

    for k, v in dic.items():
        dic_torch[target_name + "." + k] = v#torch.from_numpy(v)

    return dic_torch


def update(state_dict, new):
    for k, v in new.items():
        #if k not in state_dict:
        #    raise KeyError(k + " is not found")

        #if v.shape != state_dict[k].shape:
        #    raise ValueError(f"Shape mismatch: {v.shape} vs {state_dict[k].shape}")

        state_dict[k] = v


def discriminator_fill_statedict(statedict, vars, size):
    log_size = int(math.log(size, 2))

    update(statedict, convert_conv(vars, f"{size}x{size}/FromRGB", "convs.0"))

    conv_i = 1

    for i in range(log_size - 2, 0, -1):
        reso = 4 * 2 ** i
        update(
            statedict,
            convert_conv(vars, f"{reso}x{reso}/Conv0", f"convs.{conv_i}.conv1"),
        )
        update(
            statedict,
            convert_conv(
                vars, f"{reso}x{reso}/Conv1_down", f"convs.{conv_i}.conv2", start=1
            ),
        )
        update(
            statedict,
            convert_conv(
                vars, f"{reso}x{reso}/Skip", f"convs.{conv_i}.skip", start=1, bias=False
            ),
        )
        conv_i += 1

    update(statedict, convert_conv(vars, f"4x4/Conv", "final_conv"))
    update(statedict, convert_dense(vars, f"4x4/Dense0", "final_linear.0"))
    update(statedict, convert_dense(vars, f"Output", "final_linear.1"))

    return statedict


def fill_statedict(state_dict, vars, size, n_mlp):
    log_size = int(math.log(size, 2))

    for i in range(n_mlp):
        update(state_dict, convert_dense(vars, f"G_mapping/Dense{i}", f"style.{i + 1}"))

    update(
        state_dict,
        {
            "input.input": vars["G_synthesis/4x4/Const/const"].value().eval()
            #torch.from_numpy(
            #    vars["G_synthesis/4x4/Const/const"].value().eval()
            #)
        },
    )

    update(state_dict, convert_torgb(vars, "G_synthesis/4x4/ToRGB", "to_rgb1"))

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        update(
            state_dict,
            convert_torgb(vars, f"G_synthesis/{reso}x{reso}/ToRGB", f"to_rgbs.{i}"),
        )

    update(state_dict, convert_modconv(vars, "G_synthesis/4x4/Conv", "conv1"))

    conv_i = 0

    for i in range(log_size - 2):
        reso = 4 * 2 ** (i + 1)
        update(
            state_dict,
            convert_modconv(
                vars,
                f"G_synthesis/{reso}x{reso}/Conv0_up",
                f"convs.{conv_i}",
                flip=True,
            ),
        )
        update(
            state_dict,
            convert_modconv(
                vars, f"G_synthesis/{reso}x{reso}/Conv1", f"convs.{conv_i + 1}"
            ),
        )
        conv_i += 2

    for i in range(0, (log_size - 2) * 2 + 1):
        update(
            state_dict,
            {
                f"noises.noise_{i}": vars[f"G_synthesis/noise{i}"].value().eval()
                #torch.from_numpy(
                #    vars[f"G_synthesis/noise{i}"].value().eval()
                #)
            },
        )

    return state_dict


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Tensorflow to pytorch model checkpoint converter"
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="path to the offical StyleGAN2 repository with dnnlib/ folder",
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

    sys.path.append(args.repo)

    import dnnlib
    from dnnlib import tflib

    tflib.init_tf()

    with open(args.path, "rb") as f:
        generator, discriminator, g_ema = pickle.load(f)

    size = g_ema.output_shape[2]
    latent_avg = g_ema.vars["dlatent_avg"].value().eval()

    #batch_size = {256: 16, 512: 9, 1024: 4}
    n_sample = 1#batch_size.get(size, 25)

    z = np.random.RandomState(0).randn(n_sample, 512).astype("float32")
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.randomize_noise = False
    #img_tf = g_ema.run(z, None, **Gs_kwargs)

    n_mlp = 0
    mapping_layers_names = g_ema.__getstate__()['components']['mapping'].list_layers()
    for layer in mapping_layers_names:
        if layer[0].startswith('Dense'):
            n_mlp += 1

    state_dict = fill_statedict({}, g_ema.vars, size, n_mlp)
    state_dict['latent_avg'] = latent_avg
    state_dict['size'] = size
    state_dict['n_mlp'] = n_mlp
    #state_dict['img_tf'] = img_tf
    state_dict['n_sample'] = n_sample
    state_dict['z'] = z
    name = os.path.splitext(os.path.basename(args.path))[0]
    with open(name+"_numpy.pkl", "wb") as f:
        pickle.dump(state_dict, f)
