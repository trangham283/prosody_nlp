#!/usr/bin/env python

import os
import re
import argparse
import glob
import collections
import numpy as np
import cPickle as pickle

# constants
pitch_pov_dir = '/s0/ttmt001/speech_parsing/swbd_pitch_pov'
fbank_dir = '/s0/ttmt001/speech_parsing/swbd_fbank_energy'

OTHER = ["[silence]", "[noise]", "[laughter]", "[vocalized-noise]"]
vowels = ['aa', 'iy', 'eh', 'el', 'ah', 'ao', 'ih', 'en', 'ey', 'aw', 
'ay', 'ax', 'er','oy','ow', 'ae', 'uw']

freq_thresh = 20 # threshold for frequent words
fixed_word_length = 100
feat_dim = 6
fbank_dim = 41

split1_tails = ["ll", "m", "re", "ve", "s", "d"]
split2_tails = ["t"]

# split2 words: ending lasts 2 phones
# and anomalous words
split2_info = {
        "y'all": ["y", "all"],
        "gonna": ["gon", "na"],
        "wanna": ["wan", "na"],
        "gotta": ["got", "ta"]
        }

# split3 words: ending lasts 3 phones
split3_info = {"cannot": ["ca", "not"]}
split_set = set(split2_info.keys() + split3_info.keys())

# previous (NXT) skipped files:
# skip_files = ['3655_{A,B}', '3796_{A,B}', '3798_{A,B}', '4379_{A,B}']
# Files with switched turns
switched = set([2010, 2027, 2072, 2073, 2130, 2171, 2177, 2247, 2279, 2290, 
        2305, 2366, 2372, 2405, 2434, 2485, 2501, 2521, 2527, 2533, 2539, 
        2566, 2593, 2617, 2627, 2658, 2789, 2792, 2858, 2913, 2932, 2970, 
        3012, 3040, 3088, 3096, 3130, 3131, 3138, 3140, 3142, 3144, 3146, 
        3148, 3154,
        2006, 2064, 2110, 2235, 2262, 2292, 2303, 2339, 2476, 2514, 2543, 
        2576, 2616, 2631, 2684, 2707, 2794, 2844, 2854, 2930, 2954, 2955, 
        2960, 2963, 2968, 2981, 2983, 2994, 2999, 3000, 3013, 3018, 3039, 
        3050, 3061, 3077, 3136, 3143, 3405])

def pause2cat(p):
    if np.isnan(p):
        cat = 6
    elif p == 0.0:
        cat = 0
    elif p <= 0.05:
        cat = 1
    elif p <= 0.1:
        cat = 2
    elif p <= 0.2:
        cat = 3
    elif p <= 1.0:
        cat = 4
    else:
        cat = 5
    return cat

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

def make_array(frames):
    return np.array(frames).T

# load files of mean stats
def load_stats(stat_dir):
    head_dict = pickle.load(open(os.path.join(stat_dir, \
            'word_head_stats.pickle')))
    tail_dict = pickle.load(open(os.path.join(stat_dir, \
            'word_tail_stats.pickle')))
    word_dict = pickle.load(open(os.path.join(stat_dir, \
            'word_raw_stats.pickle')))
    phone_dict = pickle.load(open(os.path.join(stat_dir, \
            'phone_raw_stats.pickle')))
    return word_dict, phone_dict, head_dict, tail_dict

def sort_keys(pw_names):
    pw_temp = []
    for pw in pw_names:
        turn = int(pw.split('_')[0][3:])
        sent_num = int(pw.split('_')[-1][2:])
        pw_temp.append([pw, (turn, sent_num)])
    sorted_keys = sorted(pw_temp, key=lambda x: x[1])
    sorted_keys = [x[0] for x in sorted_keys]
    return sorted_keys

def preprocess_cnn_feats(file_id, speaker, file_info):
    pitch_pov_file = os.path.join(pitch_pov_dir, \
            'sw{0}-{1}.pickle'.format(file_id, speaker))
    fbank_file = os.path.join(fbank_dir, \
            'sw{0}-{1}.pickle'.format(file_id, speaker))
     
    data_pitch_pov = pickle.load(open(pitch_pov_file))
    pitch_povs = data_pitch_pov.values()[0]
             
    data_fbank = pickle.load(open(fbank_file))
    fbanks = data_fbank.values()[0]
    
    if len(pitch_povs) != len(fbanks):
        print "Length mismatch: ", len(pitch_povs), len(fbanks)
        choose_len = min(len(pitch_povs), len(fbanks))
        pitch_povs = pitch_povs[:choose_len]
        fbanks = fbanks[:choose_len]
    
    exp_fbank = np.exp(fbanks).T
    stimes = []
    etimes = []

    sorted_keys = sort_keys(file_info.keys())
    for pw in sorted_keys:
        v = file_info[pw]
        stimes.append(v['start_time'])
        etimes.append(v['end_time'])
      
    sframes = [int(np.floor(x*100)) for x in stimes]
    eframes = [int(np.ceil(x*100)) for x in etimes]
    
    # first, get voice-active regions for normalization
    turn_fb = np.empty((fbank_dim, 0))
    for i in range(len(sframes)):
        fb_frames = fbanks[sframes[i]:eframes[i]]
        # print i, np.array(fb_frames).T.shape, turn_fb.shape
        turn_fb = np.hstack([turn_fb, np.array(fb_frames).T])
    
    fbanks = np.array(fbanks).T
    e_total = fbanks[0,:]
    hi = np.max(turn_fb, 1)
    e0 = fbanks[0, :] - hi[0]
    elow = np.log(np.sum(exp_fbank[1:21,:],0)) - e_total
    ehigh = np.log(np.sum(exp_fbank[21:,:],0)) - e_total
    energy = np.array([e0, elow, ehigh])
    pitch3 = np.array(pitch_povs).T
    pitch3_energy = np.vstack([pitch3, energy])
    return pitch3_energy
   
# Get normalized word and rhyme durations
def process_ph_list(word_id, lookup, flat_phones, phone_dict):
    word_freq = lookup[word_id][0] if word_id in lookup else 0
    raw_dur = flat_phones[-1][-1] - flat_phones[0][1]
    phones = [x[0] for x in flat_phones]
    vowel_idx = [i for i,x in enumerate(phones) if x in vowels]
    if not vowel_idx: vowel_idx = [0]
    last_vowel_idx = vowel_idx[-1]
    rhyme_phones = phones[last_vowel_idx:]
    raw_rhyme_dur = flat_phones[-1][-1] - flat_phones[last_vowel_idx][1]
    mu_rhyme = sum([phone_dict[x][1] for x in rhyme_phones])
    if word_freq > freq_thresh:
        mu_word = lookup[word_id][1]
    else:
        mu_word = sum([phone_dict[x][1] for x in phones]) 
    word_norms = min(raw_dur / mu_word, 5.0)
    rhyme_norms = min(raw_rhyme_dur / mu_rhyme, 5.0)
    return word_norms, rhyme_norms

def get_word_norms(word_id, ph_list, dictionaries):
    word_dict, phone_dict, head_dict, tail_dict = dictionaries
    if len(ph_list) == 1:
        flat_phones = ph_list[0]
        wn, rn = process_ph_list(word_id, word_dict, flat_phones, phone_dict)
        word_norms = [wn]
        rhyme_norms = [rn]
    else:
        temp_splits = word_id.split("'")
        head_ph, tail_ph = ph_list
        if word_id in split2_info:
            head_id, tail_id = split2_info[word_id]
        elif word_id in split3_info:
            head_id, tail_id = split3_info[word_id]
        elif temp_splits[1] in split1_tails:
            head_id, tail_id = temp_splits
        elif temp_splits[1] in split2_tails:
            head_id = word_id[:-3]
            tail_id = word_id[-3:]
        # use word_dict to normalize heads instead of head_dict
        head_wn,head_rn = process_ph_list(head_id,word_dict,head_ph,phone_dict)
        tail_wn,tail_rn = process_ph_list(tail_id,tail_dict,tail_ph,phone_dict)
        word_norms = [head_wn, tail_wn]
        rhyme_norms = [head_rn, tail_rn]
    return word_norms, rhyme_norms

# slice speech features 
def get_word_cnns(pw, ph_range, raw_cnn_feats):
    # print pw
    start_time = ph_range[0][1]
    end_time = ph_range[-1][-1]
    start_frame = int(np.floor(start_time*100)) 
    end_frame = int(np.ceil(end_time*100))

    # empty for some reason: return all zeros
    if end_frame - start_frame <= 0:
        print pw
        return np.zeros((feat_dim, fixed_word_length))

    center_frame = int((start_frame + end_frame)/2)
    s_idx = center_frame - int(fixed_word_length/2)
    e_idx = center_frame + int(fixed_word_length/2)
    raw_word_frames = raw_cnn_feats[:, start_frame:end_frame]
    raw_count = raw_word_frames.shape[1]
    if raw_count > fixed_word_length:
        # too many frames, choose a subset
        extra_ratio = int(raw_count/fixed_word_length)
        if extra_ratio < 2: # delete things in the middle
            mask = np.ones(raw_count, dtype=bool)
            num_extra = raw_count - fixed_word_length
            not_include = range(center_frame - num_extra, \
                    center_frame + num_extra)[::2]
            # need to offset by beginning frame
            not_include = [x-start_frame for x in not_include]
            mask[not_include] = False
        else: # too big, just sample
            mask = np.zeros(raw_count, dtype=bool)
            include = range(start_frame, end_frame)[::extra_ratio]
            include = [x-start_frame for x in include]
            if len(include) > fixed_word_length: 
                # still too many frames, delete from 2 ends with skips
                num_current = len(include)
                sub_extra = num_current - fixed_word_length
                left = int(sub_extra / 2)
                right = sub_extra - left
                not_include = []
                for i in range(right):
                    idx = 2*i + 1
                    not_include.append(include[-idx])
                for i in range(left):
                    idx = 2*i 
                    not_include.append(include[idx])
                for ni in not_include:
                    include.remove(ni)
            mask[include] = True
        this_word_frames = raw_word_frames[:, mask]
    else: # not enough frames, choose frames extending from center
        this_word_frames = raw_cnn_feats[:, max(0,s_idx):e_idx]
        if s_idx < 0 and this_word_frames.shape[1] < fixed_word_length:
            this_word_frames = np.hstack([np.zeros((feat_dim,-s_idx)), \
                    this_word_frames])
        if this_word_frames.shape[1] < fixed_word_length:
            num_more = fixed_word_length - this_word_frames.shape[1]
            this_word_frames = np.hstack([this_word_frames, \
                    np.zeros((feat_dim, num_more))])
    return this_word_frames

def get_pauses(info, sorted_keys):
    pause_before = {}
    pause_after = {}
    for i, pw in enumerate(sorted_keys):
        if i == 0:
            pause_before[pw] = np.nan
        else:
            prev = sorted_keys[i-1]
            pause_before[pw] = info[pw]['start_time'] - info[prev]['end_time']
        if i == len(sorted_keys) - 1:
            pause_after[pw] = np.nan
        else:
            follow = sorted_keys[i+1]
            pause_after[pw] = info[follow]['start_time'] - info[pw]['end_time']
    return pause_before, pause_after


# NOTE: regarding switched turns:
# The way it's set up right now is actually ok
# In all udio files: 
# fbank_A contains features of left channel; fbank_B of right channel
# in info files from Vicky: text match left/right channel even though name of 
# token ids are switched
def extract_features(file_id, speaker, data_dir, out_dir, dictionaries):
    in_file = os.path.join(data_dir, \
            'word_times_sw{0}{1}.pickle'.format(file_id, speaker))
    info = pickle.load(open(in_file))
    sorted_keys = sort_keys(info.keys())

    feat_dict = collections.defaultdict(dict) 
    pause_before, pause_after = get_pauses(info, sorted_keys)
    raw_cnn_feats = preprocess_cnn_feats(file_id, speaker, info)

    for pw in sorted_keys:
        v = info[pw]
        raw_word = v['text']
        word = clean_text(raw_word)
        if 'sil' in pw or raw_word in OTHER:
            print pw, word
            exit(1)
        feat_dict[pw]['text'] = word
        feat_dict[pw]['raw_text'] = raw_word
        flat_phones = zip(v['phones'], v['phone_start_times'], \
                v['phone_end_times'])
        
        if len(flat_phones) < 2 and need_split(word): 
            print pw
            exit(1)
        
        if not need_split(word):
            ph_list = [flat_phones]
        else:
            temp_splits = word.split("'")
            if word in split2_info:
                head = flat_phones[:-2]
                tail = flat_phones[-2:]
                ph_list = [head, tail]
            elif word in split3_info:
                head = flat_phones[:-3]
                tail = flat_phones[-3:]
                ph_list = [head, tail]
            elif temp_splits[1] in split1_tails:
                head = flat_phones[:-1]
                tail = flat_phones[-1:]
                ph_list = [head, tail]
            elif temp_splits[1] in split2_tails:
                head = flat_phones[:-2]
                tail = flat_phones[-2:]
                ph_list = [head, tail]
            else:
                ph_list = [flat_phones]
       
        # DEBUG PRINTS, useful, don't remove
        #print pw, raw_word, word 
        #print "\t", [x[0] for x in ph_list[0]], [x[0] for x in ph_list[-1]]
         
        # word duration features
        word_norms, rhyme_norms = get_word_norms(word, ph_list, dictionaries)
        feat_dict[pw]['word_norm'] = word_norms
        feat_dict[pw]['rhyme_norm'] = rhyme_norms

        # pause features
        if len(ph_list)==1:
            feat_dict[pw]['pause_before'] = [pause_before[pw]]
            feat_dict[pw]['pause_after'] = [pause_after[pw]]
        else:
            feat_dict[pw]['pause_before'] = [pause_before[pw], 0.0]
            feat_dict[pw]['pause_after'] = [0.0, pause_after[pw]]
        
        # cnn features
        cnn_feats = []
        for ph_range in ph_list:
            speech_feats = get_word_cnns(pw, ph_range, raw_cnn_feats)
            assert speech_feats.shape == (feat_dim, fixed_word_length)
            cnn_feats.append(speech_feats)
        feat_dict[pw]['cnn_feats'] = cnn_feats

    pickle_file = os.path.join(out_dir, \
            'sw{0}{1}.features'.format(file_id, speaker))
    pickle.dump(feat_dict, open(pickle_file, 'w'))

if __name__ == '__main__':
    pa = argparse.ArgumentParser(description='Extract prosodic features')
    pa.add_argument('--input_dir', help='input directory', \
        default='/g/ssli/projects/disfluencies/forced_alignments')
    pa.add_argument('--data_type', help='treebank or ms', \
        default='tree_aligned')
    pa.add_argument('--output_dir', help='output directory', \
        default='/s0/ttmt001/speech_parsing/ta_features')
    pa.add_argument('--file_id', help='swbd conversation id', \
        type=int, default=2005)
    pa.add_argument('--speaker', help='speaker side', \
        type=str, default='A')
    args = pa.parse_args()

    input_dir = args.input_dir
    data_type = args.data_type
    output_dir = args.output_dir
    stat_dir = os.path.join(output_dir, 'stats', data_type)
    output_dir = os.path.join(output_dir, data_type)
    data_dir = os.path.join(input_dir, data_type)
    dictionaries = load_stats(stat_dir)

    if args.file_id != 0:
        file_id = args.file_id
        speaker = args.speaker
        extract_features(file_id, speaker, data_dir, output_dir, dictionaries)
    else:
        all_files = glob.glob(os.path.join(data_dir + '/word_times*'))
        for in_file in all_files:
            fname = os.path.basename(in_file).split('.')[0].split('_')[-1][2:]
            file_id = int(fname[:-1])
            # continuing after some bugs...
            #if file_id in skip_files: continue
            #if file_id <= 3442: continue
            speaker = fname[-1]
            print file_id, speaker
            extract_features(file_id, speaker, data_dir, output_dir, \
                    dictionaries)


