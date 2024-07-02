import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import pickle
from collections import defaultdict, Counter


def get_quantized_ab_pairs(image_lab: np.ndarray, grid: int = 10):
    ab_mat = image_lab[:, :, 1:].reshape(-1, 2)
    ab_mat_biased = ab_mat.astype(int) - 128
    ab_mat_quant = ab_mat_biased // grid * grid
    return [tuple(pair) for pair in ab_mat_quant.tolist()]


train_df = pd.read_pickle('data/ILSVRC/Metadata/train.pkl')
ab_pairs_counter = defaultdict(int)

for i in tqdm(range(len(train_df))):
    img_path = train_df.iloc[i]['image_path']

    image = cv2.imread(img_path)
    image = cv2.resize(image, (224, 224))

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

    ab_pairs_ = get_quantized_ab_pairs(image_lab)

    ab_pairs_counter_ = dict(Counter(ab_pairs_))
    for pair, count in ab_pairs_counter_.items():
        ab_pairs_counter[pair] += count


ab_pairs_counter = dict(ab_pairs_counter)
print(f"Number of unique ab pairs is: {len(ab_pairs_counter)}")


with open('data/ILSVRC/Metadata/ab_pair_counts.pkl', 'wb') as f:
    pickle.dump(ab_pairs_counter, f)
