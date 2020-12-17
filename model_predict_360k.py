import tqdm
import struct
import os
import numpy as np
import pickle
import json
import random
from collections import Counter

#from lightfm import LightFM
from scipy import sparse
from evaluate import evaluate, coverage
from implicit.als import AlternatingLeastSquares
from scipy.linalg import norm

split_folder = 'lastfm-360k'

predictions_fidelity_filename = 'predicted_features_{}.npy'
user_features_playcounts_filename = 'out_user_playcounts_als.feats'
item_features_playcounts_filename = 'out_item_playcounts_als.feats'
predictions_playcounts_filename = 'predicted_playcounts_als.npy'
gender_location = 'data/lfm-360-gender.json'

def evaluate2(iteration_tracks, items_dict, tracks_pop):
    all_songs = {}
    popularity = []
    for user in range(len(iteration_tracks)):
        if len(iteration_tracks[user]):
            curr_pop = 0
            for track in iteration_tracks[user]:
                curr_pop += tracks_pop[0, track]
                if track not in all_songs:
                    all_songs[track] = 0
                all_songs[track] += 1
            popularity.append(curr_pop/len(iteration_tracks[user]))
 
    #return len(different_songs)/len(iteration_tracks)    #return np.mean(all_songs)
    #print (len(different_songs), len(items_dict))
    #return len(different_songs)/len(items_dict)#sum(all_songs)    #return np.mean(all_songs)
    popularity = np.mean(popularity)
    different_songs = len(all_songs)
    if different_songs > len(items_dict):
        np_counts = np.zeros(different_songs, np.dtype('float64'))
    else:
        np_counts = np.zeros(len(items_dict), np.dtype('float64'))
    np_counts[:different_songs] = np.array(list(all_songs.values())) 
    return gini(np_counts), different_songs, popularity

def gini(array):
    # based on bottom eq: http://www.statsdirect.com/help/content/image/stat0206_wmf.gif
    # from: http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    array = array.flatten() #all values are treated equally, arrays must be 1d
    if np.amin(array) < 0:
        array -= np.amin(array) #values cannot be negative
    array += 0.0000001 #values cannot be 0
    array = np.sort(array) #values must be sorted
    index = np.arange(1,array.shape[0]+1) #index per array element
    n = array.shape[0]#number of array elements
    return ((np.sum((2 * index - n  - 1) * array)) / (n * np.sum(array))) #Gini coefficient


def load_feats(feat_fname, meta_only=False, nrz=False):
    with open(feat_fname, 'rb') as fin:
        keys = fin.readline().strip().split()
        R, C = struct.unpack('qq', fin.read(16))
        if meta_only:
            return keys, (R, C)
        feat = np.fromstring(fin.read(), count=R * C, dtype=np.float32)
        feat = feat.reshape((R, C))
        if nrz:
            feat = feat / np.sqrt((feat ** 2).sum(-1) + 1e-8)[..., np.newaxis]
    return keys, feat

def save(keys, feats, out_fname):
        feats = np.array(feats, dtype=np.float32)
        with open(out_fname + '.tmp', 'wb') as fout:
            fout.write(b' '.join([k.encode() for k in keys]))
            fout.write(b'\n')
            R, C = feats.shape
            fout.write(struct.pack('qq', *(R, C)))
            fout.write(feats.tostring())
        os.rename(out_fname + '.tmp', out_fname)


def train_als(impl_train_data, dims, user_ids, item_ids, user_features_file, item_features_file, save_res=True):
    model = AlternatingLeastSquares(factors=dims, iterations=50)
    model.fit(impl_train_data.T)

    user_vecs_reg = model.user_factors
    item_vecs_reg = model.item_factors
    print("USER FEAT:", user_vecs_reg.shape)
    print("ITEM FEAT:", item_vecs_reg.shape)

    if save_res==True:
        save(item_ids, item_vecs_reg, item_features_file)
        save(user_ids, user_vecs_reg, user_features_file)
    return item_ids, item_vecs_reg, user_ids, user_vecs_reg



def train(impl_train_data, dims, user_ids, item_ids, item_features_filem, user_features_file, user_features=None, save_res=True):
    model = LightFM(loss='warp', no_components=dims, max_sampled=30, user_alpha=1e-06)
    #model = model.fit(impl_train_data, epochs=50, num_threads=8)
    model = model.fit(impl_train_data, user_features=user_features, epochs=50, num_threads=8)

    user_biases, user_embeddings = model.get_user_representations(user_features)
    #user_biases, user_embeddings = model.get_user_representations()
    item_biases, item_embeddings = model.get_item_representations()
    item_vecs_reg = np.concatenate((item_embeddings, np.reshape(item_biases, (1, -1)).T), axis=1)
    user_vecs_reg = np.concatenate((user_embeddings, np.ones((1, user_biases.shape[0])).T), axis=1)
    print("USER FEAT:", user_vecs_reg.shape)
    print("ITEM FEAT:", item_vecs_reg.shape)
    if save_res==True:
        save(item_ids, item_vecs_reg, item_features_file)
        save(user_ids, user_vecs_reg, user_features_file)
    return item_ids, item_vecs_reg, user_ids, user_vecs_reg

def predict(item_vecs_reg, user_vecs_reg, prediction_file,impl_train_data, N=100, step=1000, save_res=True):
    #listened_dict = sparse.dok_matrix(impl_train_data)
    listened_dict = impl_train_data
    predicted = np.zeros((user_vecs_reg.shape[0],N), dtype=np.uint32)
    for u in range(0,user_vecs_reg.shape[0], step):
        sims = user_vecs_reg[u:u+step].dot(item_vecs_reg.T)
        curr_users = listened_dict[u:u+step].todense() == 0
        topn = np.argsort(-np.multiply(sims,curr_users), axis=1)[:,:N]
        predicted[u:u+step, :] = topn
        if u % 100000 == 0:
            print ("Precited users: ", u)
    if save_res==True:
        np.save(open(prediction_file, 'wb'), predicted)
    return predicted
from math import log2
def show_eval(predicted_x, fan_test_data,item_ids,items_gender,  sum_listen):
    topn = predicted_x.shape[1]
    print (topn)
    fan_test_data_sorted = []
    all_res = {'test_fidelity': [], 'test_engagement': [], 'test_awearnes': [], 'test_playcounts': [], 'pred_fidelity': {}, 'pred_awearnes': {}, 'pred_engagement': {}, 'pred_playcounts': {}}
    for cutoff in ('1', '3', '5', '10', '100'):
        for name in ('pred_fidelity', 'pred_awearnes', 'pred_engagement', 'pred_playcounts'):
            all_res[name][cutoff] = []

    _SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64
    artist_gender_user = []
    artist_gender_user_recommend = []
    artist_gender_dist = []
    artist_gender_first_female = []
    artist_gender_first_male = []
    for i in range(len(fan_test_data)):
        #fan_test_data_sorted.append(fan_test_data[i])
        test_u_sorted_playcount = sorted([(a, p) for a,p in fan_test_data[i]], key=lambda x: x[1])
        fan_test_data_sorted.append([a[0] for a in test_u_sorted_playcount])
        first_female = None
        first_male = None
        for p,a in enumerate(predicted_x[i]):
            if first_female == None and items_gender[a] == 'Female':
                first_female = p
            if first_male == None and items_gender[a] == 'Male':
                first_male = p
            if first_male != None and first_female != None:
                break
        if first_female != None:
            artist_gender_first_female.append(first_female)
        else:
            artist_gender_first_female.append(len(predicted_x[i])+1)
        if first_male != None:
            artist_gender_first_male.append(first_male)
        else:
            artist_gender_first_male.append(len(predicted_x[i])+1)

        listened = dict(Counter([items_gender[a[0]] for a in test_u_sorted_playcount]))
        female = 0
        male = 0
        if 'Female' in listened:
            female = listened['Female']
        if 'Male' in listened:
            male = listened['Male']
        artist_gender_user.append(female / (male+female))
        q = [female / (male+female), male/ (male+female)]
        listened= dict(Counter([items_gender[a] for a in  predicted_x[i]]))
        female = 0
        male = 0
        if 'Female' in listened:
            female = listened['Female']
        if 'Male' in listened:
            male = listened['Male']
        artist_gender_user_recommend.append(female / (male+female))
        p = [female / (male+female), male/ (male+female)]

        artist_gender_dist.append(norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2)

    print ("Distribution", np.mean(artist_gender_dist))
    print ("Female listened", np.mean(artist_gender_user))
    print ("Female recommended", np.mean(artist_gender_user_recommend))
    print ("First Female", np.mean(artist_gender_first_female))
    print ("First Male", np.mean(artist_gender_first_male))

    metrics = ['map@100', 'precision@1', 'precision@3', 'precision@5', 'precision@10', 'r-precision', 'ndcg@100']
    results = evaluate(metrics, fan_test_data_sorted, predicted_x)
    gini_val,cov_val,pop_val = evaluate2(predicted_x, item_ids, sum_listen)
    print ('FAN', results)
    print ('GINI@100', gini_val, 'pop@100', pop_val, 'coverage@100', cov_val)
    print ('Coverage@10', coverage(predicted_x.tolist(), 10), 'Coverage on FAN test set', coverage(fan_test_data_sorted, 100))
    print ('----------------------------')


def predict_pop(pop_artists, impl_train_data, N=100):
    predicted = np.zeros((impl_train_data.shape[0],N), dtype=np.uint32)
    for u in range(0, impl_train_data.shape[0]):
        curr_val = 0
        for a in pop_artists:
            if impl_train_data[u,a] == 0:
                predicted[u,curr_val] = a
                curr_val += 1
            if curr_val == 100:
               break 
    return predicted


def predict_rnd(item_ids, impl_train_data, N=100):
    predicted = np.zeros((impl_train_data.shape[0],N), dtype=np.uint32)
    items = range(len(item_ids))
    for u in range(0, impl_train_data.shape[0]):
        selected = random.sample(items, N)
        predicted[u,:] = selected
    return predicted


if __name__== "__main__":
    artists_gender = json.load(open(gender_location))
    fan_train_data = sparse.load_npz(os.path.join('data', split_folder, 'rain_data_playcount.npz')).tocsr()
    sum_listen = fan_train_data.sum(axis=0)
    fan_test_data = pickle.load(open(os.path.join('data', split_folder, 'test_data.pkl'), 'rb'))
    fan_items_dict = pickle.load(open(os.path.join('data', split_folder, 'items_dict.pkl'), 'rb'))
    items_gender = [0]*len(fan_items_dict)
    for a in fan_items_dict.keys():
        items_gender[fan_items_dict[a]] =artists_gender[a]
    fan_users_dict = pickle.load(open(os.path.join('data', split_folder,'users_dict.pkl'), 'rb'))
    print ("Item", len(fan_items_dict))
    print ("User", len(fan_users_dict))
    print (sum_listen.shape)

    model_folder = 'models'
    user_features_file = os.path.join(model_folder, split_folder, user_features_playcounts_filename)
    item_features_file = os.path.join(model_folder, split_folder, item_features_playcounts_filename)
    item_ids, item_vecs_reg, user_ids, user_vecs_reg = train_als(fan_train_data, 200, fan_users_dict, fan_items_dict, user_features_file, item_features_file, save_res=True)
    #item_ids, item_vecs_reg, user_ids, user_vecs_reg = train(fan_train_data_fidelity, 50, fan_users_dict, fan_items_dict, model_folder, save_res=True)
    #user_ids, user_vecs_reg = load_feats(user_features_file)
    #item_ids, item_vecs_reg = load_feats(item_features_file)
    predictions_file = os.path.join(model_folder, split_folder,predictions_playcounts_filename)
    #predicted = predict(item_vecs_reg, user_vecs_reg, predictions_file, fan_train_data, step=500)
    predicted = predict(item_vecs_reg, user_vecs_reg, predictions_file, fan_train_data, step=500, save_res=False)
    #predicted = np.load(predictions_file)
    print (predicted.shape, len(fan_test_data), user_vecs_reg.shape, len(user_ids))

    print ("ALS: -->")
    show_eval(predicted, fan_test_data, fan_items_dict, items_gender, sum_listen)

    print ("POP: -->")
    pop_artists = np.argsort(-sum_listen.flatten())[0,:1000].tolist()[0]
    predicted_pop = predict_pop(pop_artists, fan_train_data)
    show_eval(predicted_pop, fan_test_data, fan_items_dict, items_gender, sum_listen)

    print ("RND: -->")
    predicted_rnd = predict_rnd(fan_items_dict, fan_train_data)
    show_eval(predicted_rnd, fan_test_data, fan_items_dict, items_gender, sum_listen)


