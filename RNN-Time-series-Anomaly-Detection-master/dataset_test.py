import requests
import os
from pathlib import Path
import pickle
from shutil import unpack_archive
import numpy as np

urls = dict()
urls['ofdm'] = []
NoisePosition = np.loadtxt(
    '/home/wky/RNNAD/RNN-Time-series-Anomaly-Detection-master/dataset/ofdm/raw/NoisePosition.npy', dtype=str)
k = 0


def generate_dataset():
    for dataname in urls:
        raw_dir = Path('dataset', dataname, 'raw')
        raw_dir.mkdir(parents=True, exist_ok=True)
        for url in urls[dataname]:
            filename = raw_dir.joinpath(Path(url).name)
            print('Downloading', url)
            resp = requests.get(url)
            filename.write_bytes(resp.content)
            if filename.suffix == '':
                filename.rename(filename.with_suffix('.txt'))
            print('Saving to', filename.with_suffix('.txt'))
            if filename.suffix == '.zip':
                print('Extracting to', filename)
                unpack_archive(str(filename), extract_dir=str(raw_dir))

        for filepath in raw_dir.glob('*.txt'):
            with open(str(filepath)) as f:
                # Label anomaly points as 1 in the dataset
                labeled_data = []
                for i, line in enumerate(f):
                    tokens = [float(token) for token in line.split()]
                    if filepath.name == 'NoiseSymbol.txt':
                        if i == int(NoisePosition[k])-1:
                            tokens.append(1.0)
                            if k < len(NoisePosition)-1:
                                k += 1
                        else:
                            tokens.append(0.0)
                    labeled_data.append(tokens)
                np.savetxt('label_data.txt', labeled_data)

                # Save the labeled dataset as .pkl extension
                labeled_whole_dir = raw_dir.parent.joinpath('labeled', 'whole')
                labeled_whole_dir.mkdir(parents=True, exist_ok=True)
                with open(str(labeled_whole_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    pickle.dump(labeled_data, pkl)

                # Divide the labeled dataset into trainset and testset, then save them
                labeled_train_dir = raw_dir.parent.joinpath('labeled', 'train')
                labeled_train_dir.mkdir(parents=True, exist_ok=True)
                labeled_test_dir = raw_dir.parent.joinpath('labeled', 'test')
                labeled_test_dir.mkdir(parents=True, exist_ok=True)
                if filepath.name == 'NoiseSymbol.txt':
                    # with open(str(labeled_train_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                    #     pickle.dump(labeled_data[1000:], pkl)
                    with open(str(labeled_test_dir.joinpath(filepath.name).with_suffix('.pkl')), 'wb') as pkl:
                        pickle.dump(labeled_data, pkl)
