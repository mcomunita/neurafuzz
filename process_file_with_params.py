import argparse
import torch
import torchaudio
import os
from scipy.io.wavfile import write
from re import split as resplit
import numpy as np

import src.utils as utils
# import src.load_model as load_model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ifl', type=str, help='input file')
    parser.add_argument('--f', type=int, help='how many frames to process')
    parser.add_argument('--p', type=str, help='params values (e.g. "100, 50, 100, 75")') # BIAS=100 (fixed), GAIN, SENS=100 (fixed), REL
    parser.add_argument('--od', type=str, help='output directory')
    parser.add_argument('--mf', type=str, help='model file')

    return parser.parse_args()


def test(args):

    print(args.ifl)
    print(args.f)
    print(args.p)
    print(args.od)
    print(args.mf)

    input, sr = torchaudio.load(args.ifl, normalize=False)
    input = input[..., :args.f]
    params = [float(value) for value in args.p.split(", ")]
    params = [float(p)/100 for p in params]
    params = torch.tensor(params)

    # cuda
    if not torch.cuda.is_available():
        torch.set_default_tensor_type("torch.FloatTensor")
        device = torch.device("cpu")
        print("\ncuda device not available/not selected")
    else:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)
        print("\ncuda device available")

    model_data = utils.json_load(args.mf)
    model = utils.load_model(model_data, device=device)
    model = model.to(device)

    input = input.unsqueeze(0).to(device)
    params = params.unsqueeze(0).to(device)

    print(input.shape)
    print(params.shape)
    print(params)

    # process
    output = model.process_data(input, params)

    # filenames
    ifile = resplit("_", os.path.basename(args.ifl)[:-4])
    # pfile = f"{ifile[0]}_pred_{ifile[2]}_{args.params}.wav"
    pfile = f"{ifile[0]}_pred_{args.p}.wav"

    # save
    write(os.path.join(args.od, pfile), sr, output.cpu().numpy()[0, 0, :])


def main():
    args = parse_args()
    test(args)


if __name__ == '__main__':
    main()
