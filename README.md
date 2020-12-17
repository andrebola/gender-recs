## Description


This repository contains the code to deproduce the paper titled "Break the Loop: Gender Imbalance in Music Recommenders", by Andres Ferraro, Xavier Serra and Christine Bauer.
## Instructions

**Step 1**

Download datasets and locat them in the folder `./data`. The information of the artists should be also located in the same folder, it can be downloaded from (here)[https://zenodo.org/record/3748787].

Install dependencies specified in requirements.py

**Step 2**

Before starting to generate recommendations the data has to be processed and formatted: 

 - `python generate_mtrx.py` : Generate matrix for artists using the LFM-1b dataset
 - `python generate_mtrx_360k.py`: Generate matrix for artists using LFM-360k dataset
 - `python generate_mtrx_tracks.py`: Genrate matrix with tracks using LFM-1b dataset

**Step 3**


To run the first experiment to generate artist recommendation the following scprits must be executed:


 - `python model_predict.py` : Generate recommendations for artists using the LFM-1b dataset
 - `python model_predict_360k.py`: Generate recommendations for artists using LFM-360k dataset

To run the second experiment to generate track recommendations the following script must be executed:

 - `python model_predict_tracks.py`: Genrate recommendations with tracks using LFM-1b dataset


**Step 4**


Finally, to run the last experiment with the simulations the following script must be executed:

 - `python model_simualte_artist.py -l 0`: Generate simulation with artist recommendations using LFM-1b dataset indicating the value of lambda

## Cite


