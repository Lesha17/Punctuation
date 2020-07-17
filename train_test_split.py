import argparse
import shutil
import os
import random


def train_dev_test_split(data_dir, train_frac, dev_frac):
    textsdir = os.path.join(data_dir, 'texts')

    traindir = os.path.join(data_dir, 'train')
    os.makedirs(traindir, exist_ok=True)

    devdir = os.path.join(data_dir, 'dev')
    os.makedirs(devdir, exist_ok=True)

    testdir = os.path.join(data_dir, 'test')
    os.makedirs(testdir, exist_ok=True)
    for filename in os.listdir(textsdir):
        filepath = os.path.join(textsdir, filename)
        if os.path.isfile(filepath):
            rand_num = random.random()
            if rand_num < train_frac:
                shutil.copyfile(filepath, os.path.join(traindir, filename))
            elif rand_num < train_frac + dev_frac:
                shutil.copyfile(filepath, os.path.join(devdir, filename))
            else:
                shutil.copyfile(filepath, os.path.join(testdir, filename))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir')
    parser.add_argument('--train_frac', type=float)
    parser.add_argument('--dev_frac', type=float)
    args = parser.parse_args()

    data_dir = args.data_dir
    train_frac = args.train_frac
    dev_frac = args.dev_frac

    train_dev_test_split(data_dir, train_frac, dev_frac)
