import tqdm
import os
import random
import pickle
import json
import pandas as pd

from sklearn.model_selection import train_test_split
import numpy as np
from scipy import sparse
from collections import Counter, defaultdict

dataset_location = 'data/LFM-1b/LFM-1b_LEs.txt'
gender_location = 'data/lfm-gender.json'
random.seed(42)

def split(test_size, artists_gender):
    artists_catalog = {}
    artists_users = {}
    last_user = None
    fan_data_awe = []
    fan_data_eng = []
    fan_data_play = []
    fan_row_train = []
    fan_col_train = []
    fan_test_data = []
    test_data = []
    data_train = []
    row_train = []
    col_train = []
    fan_user_ids = []
    fan_item_ids = []
    fan_items_dict = {}
    fan_users_dict = {}
    counts_dict = {}
    user_pos = {}
    count = 0
    max_engagement = {}
    max_awearnes = {}
    track_gender = {}

    counter_tracks = [0,0]
    for line in tqdm.tqdm(open(dataset_location)):
        hists = line.strip().split('\t')
        user_pos[hists[0]] = count
        if hists[1] in artists_gender:
            artists_catalog[hists[3]] = hists[1]
            if hists[3] not in artists_users:
                artists_users[hists[3]] = set()
            artists_users[hists[3]].add(hists[0])
        count += 1

    for t in artists_catalog.keys():
        track_gender[t] = artists_gender[artists_catalog[t]]
        gender = artists_gender[artists_catalog[t]]
        if gender == "Male":
            counter_tracks[0] +=1
        elif gender == "Female":
            counter_tracks[1] +=1
    print ("Male tracks", counter_tracks[0])
    print ("Female tracks", counter_tracks[1])
    count = 0
    for line in tqdm.tqdm(open(dataset_location)):
        hists = line.strip().split('\t')
        if hists[0] not in counts_dict:
            counts_dict[hists[0]] = {}
        if hists[3] not in counts_dict[hists[0]]:
            counts_dict[hists[0]][hists[3]] =  0
        counts_dict[hists[0]][hists[3]] += 1
        last_user = hists[0]
        if user_pos[last_user] == count:
            counts = counts_dict[last_user]
            artist_fan = []
            for t in counts.keys():
                if  t not in artists_catalog or (artists_catalog[t] not in artists_gender) or len(artists_users[t]) < 30:
                    continue
                total_tracks_listen = counts[t]
                artist_fan.append((t, total_tracks_listen))
                track_gender[t] = artists_gender[artists_catalog[t]]
            if len(artist_fan) <= 10:
                count +=1
                del counts_dict[last_user]
                continue
            del counts_dict[last_user]

            artist_fan_dict = {a:1 for a in artist_fan}
            if last_user in fan_users_dict:
                print ("PROBLEM!!!!")
            fan_users_dict[last_user] = len(fan_user_ids)
            fan_user_ids.append(last_user)
            random.shuffle(artist_fan)
            split = round(len(artist_fan)*test_size)
            train_u = artist_fan[split:]
            test_u = artist_fan[:split]
            for item, play in train_u:
                if item not in fan_items_dict:
                    fan_items_dict[item] = len(fan_item_ids)
                    fan_item_ids.append(item)
                fan_col_train.append(fan_items_dict[item])
                fan_row_train.append(fan_users_dict[last_user])
                fan_data_play.append(play)
            #test_u_sorted = sorted([(a,v,p) for a,v,p in test_u], key=lambda x: x[1])
            fan_test_u = []
            for item, play in test_u:
                if item not in fan_items_dict:
                    fan_items_dict[item] = len(fan_item_ids)
                    fan_item_ids.append(item)
                fan_test_u.append((fan_items_dict[item], play))
            fan_test_data.append(fan_test_u)
        count += 1
    listened = dict(Counter(track_gender.values()))
    print ("Dataset gender", listened)

    return fan_data_play, fan_row_train, fan_col_train, fan_test_data, fan_items_dict, fan_users_dict, track_gender


if __name__== "__main__":
    artists_gender = json.load(open(gender_location))

    fan_data_play, fan_row_train, fan_col_train, fan_test_data, fan_items_dict, fan_users_dict, track_gender= split(0.2, artists_gender)

    json.dump(track_gender, open(os.path.join('data', 'lastfm', 'track_gender.json'), 'w'))
    fan_train_play = sparse.coo_matrix((fan_data_play, (fan_row_train, fan_col_train)), dtype=np.float32)
    #print ("TRAIN USERS", fan_train_play.shape)
    sparse.save_npz(os.path.join('data', 'lastfm', 'tracks_train_data_playcount.npz'), fan_train_play)
    pickle.dump(fan_test_data, open(os.path.join('data', 'lastfm','tracks_test_data.pkl'), 'wb'))
    pickle.dump(fan_items_dict, open(os.path.join('data','lastfm', 'tracks_items_dict.pkl'), 'wb'))
    pickle.dump(fan_users_dict, open(os.path.join('data','lastfm', 'tracks_users_dict.pkl'), 'wb'))
