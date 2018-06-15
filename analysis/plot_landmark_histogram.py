# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import matplotlib.pyplot as plt

from ttw.data_loader import Map

plt.switch_backend('agg')
plt.style.use('ggplot')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='./data', help='Path to talkthewalk dataset')

    args = parser.parse_args()

    neighborhoods = ['fidi', 'hellskitchen', 'williamsburg', 'uppereast', 'eastvillage']
    map = Map(args.data_dir, neighborhoods)

    num_examples = [0]*len(map.landmark_dict)
    labels = [map.landmark_dict.decode(i) for i in range(len(map.landmark_dict))]
    for neighborhood in neighborhoods:
        for x in map.coord_to_landmarks[neighborhood]:
            for y in x:
                for l in y:
                    num_examples[l] += 1
    labels = [l for i, l in enumerate(labels) if num_examples[i] > 0]
    num_examples = [x for x in num_examples if x > 0]

    plt.xticks(rotation='vertical')
    plt.bar(labels, num_examples, color='royalblue')
    plt.ylabel('Number of')
    plt.tight_layout()
    plt.savefig('landmark_histogram.png')
