import tqdm
import struct
import os
import numpy as np
import pickle
import json
import random
import argparse
from collections import Counter

#from lightfm import LightFM
from scipy import sparse
from evaluate import evaluate, coverage
from implicit.als import AlternatingLeastSquares
from scipy.linalg import norm

os.environ["OPENBLAS_NUM_THREADS"] = "1"
split_folder = 'lastfm'

user_features_playcounts_filename = 'out_user_playcounts_als.feats'
item_features_playcounts_filename = 'out_item_playcounts_als.feats'
predictions_playcounts_filename = 'predicted_playcounts_als.npy'
gender_location = 'data/lfm-gender.json'

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
    #print("USER FEAT:", user_vecs_reg.shape)
    #print("ITEM FEAT:", item_vecs_reg.shape)
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
        #if u % 100000 == 0:
        #    print ("Precited users: ", u)
    if save_res==True:
        np.save(open(prediction_file, 'wb'), predicted)
    return predicted

def rerank(predicted, items_gender, lambda1=10):
    ret_all = []
    ret_non_zero = []
    zero_users = 0
    for u in range(0,predicted.shape[0]):
        counter = 0
        recs_dict = {item:p for p,item in enumerate(predicted[u, :])}
        for i, track in enumerate(recs_dict.keys()):
            if items_gender[track] == "Male":
                recs_dict[track] += lambda1
                if i< 10:
                    counter += 1
        ret_all.append(counter)
        if counter == 0:
            zero_users += 1
        else:
            ret_non_zero.append(counter)

        predicted[u] = np.array([k for k,v in sorted(recs_dict.items(), key=lambda x: x[1])])
        #if u % 50000 == 0:
        #    print ("reranked users: ", u)
    return np.mean(ret_all), np.mean(ret_non_zero), zero_users

from math import log2
def show_eval(predicted_x, fan_test_data,item_ids,items_gender,  sum_listen, changes):
    topn = predicted_x.shape[1]
    fan_test_data_sorted = []
    fan_test_data_male = []
    fan_test_data_female = []
    predicted_male = []
    predicted_female = []
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
    reco_set= {}
    reco_set_10= {}
    for i in range(len(fan_test_data)):
        #fan_test_data_sorted.append(fan_test_data[i])
        test_u_sorted_playcount = sorted([(a, p) for a,p in fan_test_data[i]], key=lambda x: x[1])
        fan_test_data_sorted.append([a[0] for a in test_u_sorted_playcount])
        fan_test_data_male.append([a[0] for a in test_u_sorted_playcount if items_gender[a[0]] == "Male"])
        fan_test_data_female.append([a[0] for a in test_u_sorted_playcount if items_gender[a[0]] == "Female"])
        if len(fan_test_data_sorted) == 0:
            continue
        first_female = None
        first_male = None
        curr_predict_female = []
        curr_predict_male = []
        for p,a in enumerate(predicted_x[i]):
            if first_female == None and items_gender[a] == 'Female':
                first_female = p
            if first_male == None and items_gender[a] == 'Male':
                first_male = p
            #if first_male != None and first_female != None:
            #    break
            if items_gender[a] == 'Female':
                curr_predict_female.append(a)
            elif items_gender[a] == 'Male':
                curr_predict_male.append(a)
        predicted_female.append(curr_predict_female)
        predicted_male.append(curr_predict_male)
        if first_female != None:
            artist_gender_first_female.append(first_female)
        else:
            artist_gender_first_female.append(len(predicted_x[i])+1)
        if first_male != None:
            artist_gender_first_male.append(first_male)
        else:
            artist_gender_first_male.append(len(predicted_x[i])+1)

        reco_set.update({a:1 for a in  predicted_x[i]})
        reco_set_10.update({a:1 for a in  predicted_x[i][:10]})

        listened_gender = None
        listened = dict(Counter([items_gender[a[0]] for a in test_u_sorted_playcount]))
        female = 0
        male = 0
        if 'Female' in listened:
            female = listened['Female']
        if 'Male' in listened:
            male = listened['Male']
        if (male+female) > 0:
            #artist_gender_user.append(female / (male+female))
            listened_gender = female / (male+female)
            q = [female / (male+female), male/ (male+female)]

        listened= dict(Counter([items_gender[a] for a in  predicted_x[i]]))
        female = 0
        male = 0
        if 'Female' in listened:
            female = listened['Female']
        if 'Male' in listened:
            male = listened['Male']
        if (male+female) > 0 and listened_gender != None:
            artist_gender_user_recommend.append(female / (male+female))
            p = [female / (male+female), male/ (male+female)]

            artist_gender_user.append(listened_gender)
            artist_gender_dist.append(norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2)

    reco_set_total = dict(Counter([items_gender[a] for a in reco_set.keys()]))
    reco_set_10_total = dict(Counter([items_gender[a] for a in reco_set_10.keys()]))
    header = 'Coverage@100 Male, Coverage@100 Female, Coverage@10 Male, Coverage@10 Female, Distribution, Female listened, Female recommended, First Female, First Male'
    res = []
    res.append(reco_set_total['Male'])
    res.append(reco_set_total['Female'])
    res.append(reco_set_10_total['Male'])
    res.append(reco_set_10_total['Female'])
    res.append(np.mean(artist_gender_dist))
    res.append(np.mean(artist_gender_user))
    res.append(np.mean(artist_gender_user_recommend))
    res.append(np.mean(artist_gender_first_female))
    res.append(np.mean(artist_gender_first_male))

    header += ', GINI@100, pop@100, coverage@100, Coverage@10, Coverage on FAN test set@10, all changes, non_zero changes, zero_users, iter'
    gini_val,cov_val,pop_val = evaluate2(predicted_x, item_ids, sum_listen)

    res.append(gini_val)
    res.append(pop_val)
    res.append(cov_val)
    res.append(coverage(predicted_x.tolist(), 10))
    res.append(coverage(fan_test_data_sorted, 100))
    res.append(changes[0])
    res.append(changes[1])
    res.append(changes[2])
    res.append(changes[3])
    print (header)
    for i in range(len(res)):
        if i in (0,1,2,3,10,11,12,13,16,17):
            print(int(res[i]),end=', ')
        else:
            print('{:.4f}'.format(res[i]),end=', ')
    print()

    metrics = ['map@10', 'precision@1', 'precision@3', 'precision@5', 'precision@10', 'r-precision', 'ndcg@10']
    results = evaluate(metrics, fan_test_data_sorted, predicted_x)#[:, :10])
    print_head = ''
    print_str = ''
    for metric in metrics:
       print_head += metric +", "

    for metric in metrics:
        print_str+= ', {:.4f}'.format(results[metric])
    print (print_head)
    print (print_str)

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
    parser = argparse.ArgumentParser(description='Run model training and evaluation.')
    parser.add_argument('-l', "--lambda1", default='0')
    args = parser.parse_args()
    lambda1 = int(args.lambda1)


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
    dims = 300
    user_features_file = os.path.join(model_folder, split_folder, user_features_playcounts_filename.format(dims))
    item_features_file = os.path.join(model_folder, split_folder, item_features_playcounts_filename.format(dims))
    item_ids, item_vecs_reg, user_ids, user_vecs_reg = train_als(fan_train_data, dims, fan_users_dict, fan_items_dict, user_features_file, item_features_file, save_res=True)
    #item_ids, item_vecs_reg, user_ids, user_vecs_reg = train(fan_train_data_fidelity, 50, fan_users_dict, fan_items_dict, model_folder, save_res=True)
    #user_ids, user_vecs_reg = load_feats(user_features_file)
    #item_ids, item_vecs_reg = load_feats(item_features_file)
    predictions_file = os.path.join(model_folder, split_folder,predictions_playcounts_filename.format(dims))
    predicted = predict(item_vecs_reg, user_vecs_reg, predictions_file, fan_train_data, step=500)
    #predicted = np.load(predictions_file)
    #rerank(predicted, items_gender, lambda1)
    print (predicted.shape, len(fan_test_data), user_vecs_reg.shape, len(user_ids))
    #print ("ALS: -->", dims, "Lambda", lambda1)
    #show_eval(predicted, fan_test_data, fan_items_dict, items_gender, sum_listen)

    N = 100
    step = 2000
    for iter_n in range(21):
        artists_count = Counter()
        predicted = np.zeros((user_vecs_reg.shape[0],N), dtype=np.uint32)
        for u in range(0,user_vecs_reg.shape[0],step):#len(user_ids)):
            sims = user_vecs_reg[u:u+step].dot(item_vecs_reg.T)
            topn = np.argsort(-sims, axis=1)[:,:N]#.flatten()
            #curr_users = fan_train_data[u:u+step].todense() == 0
            #topn = np.argsort(-np.multiply(sims,curr_users), axis=1)[:,:N]
            predicted[u:u+step, :] = topn

        changes = rerank(predicted, items_gender, lambda1)
        changes = list(changes)

        M = 10
        for u in range(0,user_vecs_reg.shape[0],step):#len(user_ids)):
            topn = predicted[u:u+step, :][:, :M].flatten()
            u_min = min(u+step, user_vecs_reg.shape[0])
            rows = np.repeat(np.arange(u,u_min), M)
            mtrx_sum = sparse.csr_matrix((np.repeat(M,topn.shape[0]), (rows, topn)),shape=fan_train_data.shape, dtype=np.float32)
            fan_train_data = fan_train_data+mtrx_sum
            artists_count.update(topn.tolist())
        n_artists = len(artists_count)
        np_counts = np.zeros(item_vecs_reg.shape[0], np.dtype('float64'))
        np_counts[:n_artists] = np.array(list(artists_count.values())) 
        #pickle.dump(artists_count, open("data/artists_count_{}.pkl".format(str(iter_n)),"wb"))
        #print ("iter:", str(iter_n))
        #print (artists_count.most_common(10))
        #print ("coverage:", n_artists)
        changes.append(iter_n)
        show_eval(predicted, fan_test_data, fan_items_dict, items_gender, sum_listen, changes)

        if iter_n % 10 == 0:
            #out_item_features = 'data/items_{}_{}.feat'.format(dimms,str(iter_n))
            #out_user_features = 'data/users_{}_{}.feat'.format(dimms,str(iter_n))
            #user_features_file = os.path.join(model_folder, split_folder, user_features_playcounts_filename.format(dims))
            #item_features_file = os.path.join(model_folder, split_folder, item_features_playcounts_filename.format(dims))
            item_ids, item_vecs_reg, user_ids, user_vecs_reg = train_als(fan_train_data, 300, fan_users_dict, fan_items_dict, user_features_file, item_features_file, save_res=False)
        else:
            item_ids, item_vecs_reg, user_ids, user_vecs_reg = train_als(fan_train_data, 300, fan_users_dict, fan_items_dict, user_features_file, item_features_file, save_res=False)
 
