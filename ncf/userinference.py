#!/usr/bin/env python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import sys
import torch.jit
import time
from argparse import ArgumentParser
import numpy as np
import torch

from neumf import NeuMF


def parse_args():
    parser = ArgumentParser(
        description="Benchmark inference performance of the NCF model"
    )
    parser.add_argument(
        "model_checkpoint_path",
        type=str,
        help="Path to the checkpoint file to be loaded before training/evaluation",
    )

    parser.add_argument(
        "movie_db_file",
        type=str,
        help="Path to the movies.csv file from the training dataset",
    )

    parser.add_argument(
        "--output-dir", type=str, default="./", help="Directory to save output file"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    state_dict = torch.load(args.model_checkpoint_path, map_location=torch.device("cpu"))

    layers = [state_dict["mlp.0.weight"].size()[1]]
    layer_keys = sorted(
        [k for k in state_dict.keys() if (k.startswith("mlp") and k.endswith("bias"))]
    )
    layers.extend([state_dict[k].size()[0] for k in layer_keys])

    n_items = state_dict["mf_item_embed.weight"].size()[0]
    n_users = state_dict["mf_user_embed.weight"].size()[0]

    model = NeuMF(
        nb_users=n_users,
        nb_items=n_items,
        mf_dim=state_dict["mf_item_embed.weight"].size()[1],
        mlp_layer_sizes=layers,
        dropout=0.5,
    )

    model.load_state_dict(state_dict)

    model.eval()

    users = torch.LongTensor(np.full(n_items, n_users - 1))
    items = torch.LongTensor(np.arange(n_items, dtype=np.int64))

    predictions = model(users, items, sigmoid=True)

    predictions = predictions.detach().numpy().squeeze()

    if args.output_dir != "./":
        try:
            os.makedirs(args.output_dir, exist_ok=True)
        except OSError as err:
            print("Failed to create output directory: {}".format(err))
            sys.exit(-1)

    names = []
    with open(args.movie_db_file) as fh:
        for line in fh:
            mid, _, name = line.partition(",")
            name, _, tail = name.rpartition(",")
            name = name.strip('"')
            names.append(name)

    predictions_file=os.path.join(args.output_dir, "predictions.csv")
    with open(predictions_file, "wt") as fh:
        fh.write("Movie, Predicted Rating\n")
        argsorts = np.argsort(predictions)[::-1]
        for idx in argsorts:
            fh.write("{}, {:0.2f}\n".format(names[idx], predictions[idx] * 5))

    print("Predictions saved to {}".format(predictions_file))

    return


if __name__ == "__main__":
    main()
