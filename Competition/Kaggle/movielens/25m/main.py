# coding:utf-8
# https://www.kaggle.com/jamesloy/deep-learning-based-recommender-systems
from torch.utils.data import Dataset
from models import NCF
import torch
import warnings
import numpy as np
import pandas as pd
from tqdm.notebook import tqdm
import pytorch_lightning as pl
import datetime

warnings.filterwarnings("always")

rating_path = r'E:\DataSet\DataSet\RecommendationSystem\MovieLens\ml-25m\ratings.csv'


def date_parse(time_in_secs):
    return datetime.datetime.fromtimestamp(float(time_in_secs))


# ratings = pd.read_csv(filepath_or_buffer=rating_path, parse_dates=True,
#                       date_parser=date_parse, index_col='timestamp')
ratings = pd.read_csv(filepath_or_buffer=rating_path, parse_dates=['timestamp'])
ratings['timestamp'] = pd.to_datetime(ratings['timestamp'], unit='s')

rand_userIds = np.random.choice(ratings['userId'].unique(),
                                size=int(len(ratings['userId'].unique()) * 0.3),
                                replace=False)

ratings = ratings.loc[ratings['userId'].isin(rand_userIds)]

print('There are {} rows of data from {} users'.format(len(ratings), len(rand_userIds)))
print(ratings.sample(5))

# raise DataError("No numeric types to aggregate")
# pandas.core.base.DataError: No numeric types to aggregate
# ratings['userId'] = pd.to_numeric(ratings['userId'])
ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)

# ratings['rank_latest'] = ratings.groupby(['userId'])['timestamp'].rank(method='first', ascending=False)
print(ratings.sample(5))

train_ratings = ratings[ratings['rank_latest'] != 1]
test_ratings = ratings[ratings['rank_latest'] == 1]

# drop columns that we no longer need
train_ratings = train_ratings[['userId', 'movieId', 'rating']]
test_ratings = test_ratings[['userId', 'movieId', 'rating']]

train_ratings.loc[:, 'rating'] = 1
train_ratings.sample(5)

# Get a list of all movie IDs
all_movieIds = ratings['movieId'].unique()

# Placeholders that will hold the training data
users, items, labels = [], [], []

# This is the set of items that each user has interaction with
user_item_set = set(zip(train_ratings['userId'], train_ratings['movieId']))

# 4:1 ratio of negative to positive samples
num_negatives = 4

for (u, i) in tqdm(user_item_set):
    users.append(u)
    items.append(i)
    labels.append(1)  # items that the user has interacted with are positive
    for _ in range(num_negatives):
        # randomly select an item
        negative_item = np.random.choice(all_movieIds)
        # check that the user has not interacted with this item
        while (u, negative_item) in user_item_set:
            negative_item = np.random.choice(all_movieIds)
        users.append(u)
        items.append(negative_item)
        labels.append(0)  # items not interacted with are negative

num_users = ratings['userId'].max() + 1
num_items = ratings['movieId'].max() + 1

all_movieIds = ratings['movieId'].unique()

model = NCF(num_users, num_items, train_ratings, all_movieIds)

trainer = pl.Trainer(max_epochs=5, gpus=1, reload_dataloaders_every_epoch=True, progress_bar_refresh_rate=50,
                     logger=False, checkpoint_callback=False)
trainer.fit(model)

# User-item pairs for testing
test_user_item_set = set(zip(test_ratings['userId'], test_ratings['movieId']))

# Dict of all items that are interacted with by each user
user_interacted_items = ratings.groupby('userId')['movieId'].apply(list).to_dict()

hits = []
for (u, i) in tqdm(test_user_item_set):
    interacted_items = user_interacted_items[u]
    not_interacted_items = set(all_movieIds) - set(interacted_items)
    selected_not_interacted = list(np.random.choice(list(not_interacted_items), 99))
    test_items = selected_not_interacted + [i]

    predicted_labels = np.squeeze(model(torch.tensor([u] * 100),
                                        torch.tensor(test_items)).detach().numpy())

    top10_items = [test_items[i] for i in np.argsort(predicted_labels)[::-1][0:10].tolist()]

    if i in top10_items:
        hits.append(1)
    else:
        hits.append(0)

print("The Hit Ratio @ 10 is {:.2f}".format(np.average(hits)))
