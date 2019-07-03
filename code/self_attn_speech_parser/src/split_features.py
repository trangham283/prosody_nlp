import os
import sys
import glob
import pandas as pd
import argparse
import numpy as np
import cPickle as pickle

data_dir = "/g/ssli/data/CTS-English/swbd_align/seq2seq_features"
output_dir = "/s0/ttmt001/speech_parsing/features"

def extract_features(split):
    dur_dict = {}
    pause_dict = {}
    pcoef_dict = {}
    pitch_dict = {}
    mfcc_dict = {}
    fbank_dict = {}
    partition_dict = {}
    data_file = os.path.join(data_dir, split + '_mep.pickle')
    data = pickle.load(open(data_file))
    data = [item for sublist in data for item in sublist]
    for sent_id, features, parse_ids in data:
        dur_dict[sent_id] = np.vstack([np.array(features['dur_stats']), \
                np.array(features['word_dur'])])
        pause_dict[sent_id] = {}
        pause_dict[sent_id]['pause_bef'] = features['pause_bef']
        pause_dict[sent_id]['pause_aft'] = features['pause_aft']
        pcoef_dict[sent_id] = np.array(features['pitch_stats']).T
        partition_dict[sent_id] = features['speech_frames']['partition']
        frames = features['speech_frames']['frames']
        # frames = np.vstack([pitch3, mfccs, efeats])
        pitch_dict[sent_id] = frames[:3, :]
        mfcc_dict[sent_id] = frames[3:-3, :]
        fbank_dict[sent_id] = frames[-3:, :]
    dur_name = os.path.join(output_dir, split + '_duration.pickle')
    pause_name = os.path.join(output_dir, split + '_pause.pickle')
    pcoef_name = os.path.join(output_dir, split + '_f0coefs.pickle')
    partition_name = os.path.join(output_dir, split + '_partition.pickle')
    pitch_name = os.path.join(output_dir, split + '_pitch.pickle')
    fbank_name = os.path.join(output_dir, split + '_fbank.pickle')
    mfcc_name = os.path.join(output_dir, split + '_mfcc.pickle')
    pickle.dump(dur_dict, open(dur_name, 'w'))
    pickle.dump(pause_dict, open(pause_name, 'w'))
    pickle.dump(partition_dict, open(partition_name, 'w'))
    pickle.dump(pcoef_dict, open(pcoef_name, 'w'))
    pickle.dump(pitch_dict, open(pitch_name, 'w'))
    pickle.dump(mfcc_dict, open(mfcc_name, 'w'))
    pickle.dump(fbank_dict, open(fbank_name, 'w'))


pa = argparse.ArgumentParser(description='extract features')
pa.add_argument('--split', type=str, default='dev')
args = pa.parse_args()
split = args.split
extract_features(split)



