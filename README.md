## Description


This repository contains the code to reproduce the results of the paper entitled "Break the Loop: Gender Imbalance in Music Recommenders", by Andres Ferraro, Xavier Serra, and Christine Bauer.

## Instructions

**Step 1**

Download the datasets and locate them in the folder `./data`. The artist information should be also located in the same folder; it can be downloaded from [here](https://zenodo.org/record/3748787).

Install dependencies specified in requirements.txt

**Step 2**

Before starting to generate recommendations the data has to be processed and formatted: 

 - `python generate_mtrx.py` : Generate matrix for artists using the LFM-1b dataset
 - `python generate_mtrx_360k.py`: Generate matrix for artists using LFM-360k dataset
 - `python generate_mtrx_tracks.py`: Genrate matrix with tracks using LFM-1b dataset

**Step 3**

To run the first experiment (generate artist recommendations) the following scprits must be executed:

 - `python model_predict.py` : Generate recommendations for artists using the LFM-1b dataset
 - `python model_predict_360k.py`: Generate artist recommendations using the LFM-360k dataset

To run the second experiment (generate track recommendations) the following script must be executed:

 - `python model_predict_tracks.py`: Genrate track recommendations using the LFM-1b dataset

**Step 4**

Finally, to run the last experiment (simulations) the following script must be executed:

 - `python model_simualte_artist.py -l 0`: Generate simulation with artist recommendations using LFM-1b dataset indicating the value of lambda

## Cite

Andrés Ferraro, Xavier Serra, and Christine Bauer (2021). Break the Loop: Gender Imbalance in Music Recommenders. In Proceedings of the 2021 ACM SIGIR Conference on Human Information Interaction and Retrieval (CHIIR ’21), March 14–19, 2021, Canberra, ACT, Australia. ACM, New York, NY, USA. https://doi.org/10.1145/3406522.3446033
