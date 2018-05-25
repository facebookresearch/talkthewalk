import os

import matplotlib.pyplot as plt
import numpy

from ttw.data_loader import ResnetFeatures, Map
from ttw.train.classify_landmarks import load_data

plt.switch_backend('agg')
plt.style.use('ggplot')

if __name__ == '__main__':
    neighborhoods = ['fidi', 'hellskitchen', 'williamsburg', 'uppereast', 'eastvillage']
    landmarks = Map(neighborhoods)
    data_dir = './data'

    feature_loaders = dict()
    feature_loaders['resnet'] = ResnetFeatures(os.path.join(data_dir, 'resnetfeat.json'), pca=False, n_components=None)

    assert (len(feature_loaders) > 0)

    Xs, ys = load_data(neighborhoods, feature_loaders)
    ys = numpy.array(ys)

    num_examples = ys.sum(axis=0)
    labels = landmarks.itos

    plt.xticks(rotation='vertical')
    plt.bar(labels, num_examples, color='royalblue')
    plt.ylabel('Number of')
    plt.tight_layout()
    plt.savefig('landmark_histogram.png')