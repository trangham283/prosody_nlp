#!/usr/bin/env python

import os
import re
import argparse
import glob
import collections
import numpy as np
import cPickle as pickle

split1_tails = ["ll", "m", "re", "ve", "s"]
split2_tails = ["t"]

# split2 words: ending lasts 2 phones
# and anomalous words
split2_info = {
        "y'all": ["y", "all"],
        "gonna": ["gon", "na"],
        "wanna": ["wan", "na"]
        }

# split3 words: ending lasts 3 phones
split3_info = {"cannot": ["ca", "not"]}
split_set = set(split2_info.keys() + split3_info.keys())

vowels = ['aa', 'iy', 'eh', 'el', 'ah', 'ao', 'ih', 'en', 'ey', 'aw', 
'ay', 'ax', 'er','oy','ow', 'ae', 'uw']

def process_head_tail_cases(info):
    word = info['text'].lower()
    word = clean_text(word)
    temp_splits = word.split("'")
    phones = info['phones']
    phone_ends = info['phone_end_times']
    phone_starts = info['phone_start_times']
    if word in split3_info:
        head_time = phone_ends[:-3][-1] - phone_starts[:-3][0] 
        tail_time = phone_ends[-3:][-1] - phone_starts[-3:][0]
        head_id, tail_id = split3_info[word]
    elif word in split2_info:
        head_time = phone_ends[:-2][-1] - phone_starts[:-2][0] 
        tail_time = phone_ends[-2:][-1] - phone_starts[-2:][0]
        head_id, tail_id = split2_info[word]
    #if word in split1_info:
    elif temp_splits[1] in split1_tails:
        head_time = phone_ends[:-1][-1] - phone_starts[:-1][0] 
        tail_time = phone_ends[-1:][-1] - phone_starts[-1:][0]
        #head_id = split1_info[word][0]
        #tail_id = split1_info[word][-1]
        head_id, tail_id = temp_splits
    elif temp_splits[1] in split2_tails:
        head_time = phone_ends[:-2][-1] - phone_starts[:-2][0] 
        tail_time = phone_ends[-2:][-1] - phone_starts[-2:][0]
        head_id = word[:-3]
        tail_id = word[-3:]
    return head_id, head_time, tail_id, tail_time

def need_split(word):
    if word in split_set: return True
    if "'" in word:
        temp_splits = word.split("'")
        if len(temp_splits) <= 1 or not temp_splits[0]:
            return False
        if temp_splits[0] == "-":
            return False
        if temp_splits[1] in split1_tails:
            return True
        if temp_splits[1] in split2_tails and temp_splits[0][-1]=='n':
            if word not in ["-n't", "n't"]: 
                return True
    return False

# Clean MS token to have "written" form
def clean_text(raw_word):
    word = raw_word.lower()
    word = word.replace("_1", "")
    if '[laughter-' in word:
        word = word.lstrip('[laughther').rstrip(']').lstrip('-')
    if "/" in word and "[" in word and "]" in word:
        word = word.split("/")[-1].replace("]", "")
    if "[" in word:
        word = re.sub(r'\[[^)]*\]','', word)
    if "{" in word and "}" in word:
        word = word.replace("{", "").replace("}", "")
    return word

# get data stats from training set
def get_data_stats(data_dir, stat_dir):
    # files of mean stats
    head_dict_file = os.path.join(stat_dir, 'word_head_stats.pickle')
    tail_dict_file = os.path.join(stat_dir, 'word_tail_stats.pickle')
    word_dict_file = os.path.join(stat_dir, 'word_raw_stats.pickle')
    phone_dict_file = os.path.join(stat_dir, 'phone_raw_stats.pickle')

    head_dict = {}
    tail_dict = {}
    word_dict = {}
    phone_dict = {}

    file_list = glob.glob(data_dir + '/word_times_sw3*.pickle') + \
            glob.glob(data_dir + '/word_times_sw2*.pickle')
    for f in file_list:
        data = pickle.load(open(f))
        for k in sorted(data.keys()):
            info = data[k]
            raw_word = info['text'].lower()
            if "[" in raw_word or "_1" in raw_word: continue
            word = clean_text(raw_word)
            # ignore anomalous pronunciations for purposes of getting stas
            phones = info['phones']
            word_dur = info['end_time'] - info['start_time']
            if word not in word_dict: word_dict[word] = []
            if word_dur > 0:
                # ignore weird cases without duration in stats
                word_dict[word].append(word_dur)
            for i, p in enumerate(phones):
                phone_dur = info['phone_end_times'][i] \
                        - info['phone_start_times'][i]
                if p not in phone_dict: phone_dict[p] = []
                if phone_dur > 0:
                    phone_dict[p].append(phone_dur)
            if need_split(word):
                print f, k, raw_word, word
                h_id, h_time, t_id, t_time = process_head_tail_cases(info)
                if h_id not in head_dict: head_dict[h_id] = []
                if h_time > 0: head_dict[h_id].append(h_time)
                if t_id not in tail_dict: tail_dict[t_id] = []
                if t_time > 0: tail_dict[t_id].append(t_time)

    word_stats = {}
    for k, v in word_dict.iteritems():
        word_stats[k] = [len(v), np.mean(v), np.std(v)]
    pickle.dump(word_stats, open(word_dict_file, 'w'))

    phone_stats = {}
    for k, v in phone_dict.iteritems():
        phone_stats[k] = [len(v), np.mean(v), np.std(v)]
    pickle.dump(phone_stats, open(phone_dict_file, 'w'))

    head_stats = {}
    for k, v in head_dict.iteritems():
        head_stats[k] = [len(v), np.mean(v), np.std(v)]
    pickle.dump(head_stats, open(head_dict_file, 'w'))

    tail_stats = {}
    for k, v in tail_dict.iteritems():
        tail_stats[k] = [len(v), np.mean(v), np.std(v)]
    pickle.dump(tail_stats, open(tail_dict_file, 'w'))

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Get stats on durations')
    pa.add_argument('--input_dir', help='input directory', \
        default='/g/ssli/projects/disfluencies/forced_alignments')
    pa.add_argument('--data_type', help='treebank or ms', \
        default='tree_aligned')
    pa.add_argument('--output_dir', help='output directory', \
        default='/s0/ttmt001/speech_parsing/ta_features')
    args = pa.parse_args()

    input_dir = args.input_dir
    data_type = args.data_type
    output_dir = args.output_dir

    data_dir = os.path.join(input_dir, data_type)
    stat_dir = os.path.join(output_dir, 'stats', data_type)
    get_data_stats(data_dir, stat_dir)
