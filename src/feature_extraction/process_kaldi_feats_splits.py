#!/user/bin/env python

import os
import sys
import pandas
import argparse
import numpy as np
import cPickle as pickle
import glob


nsplit = 4 # number of splits when kaldi was called

def process_feats(args):
    in_dir = args.in_dir
    out_dir = args.out_dir
    split = str(args.split)
    feattype = args.feattype
    raw_dir = os.path.join(in_dir, 'sph_splits', 'sph' + split, feattype)
    prefix = 'swbd_' + feattype
    output_dir = os.path.join(out_dir, prefix)
    print raw_dir
    print output_dir

    done_files = glob.glob(output_dir + '*')
    done_files = [os.path.basename(x).split('.')[0] for x in done_files]
    if feattype == 'pitch_pov':
        numc = 3
    elif feattype == 'mfcc':
        numc = 13
    else:
        numc = 41
    for i in range(1, nsplit+1):
        suffix = feattype.split('_')[0]
        raw_file = os.path.join(raw_dir, 'raw_%s_audio.%d.txt' %(suffix,i) )
        raw_lines = open(raw_file).readlines()
        sindices = [i for i,x in enumerate(raw_lines) if 'sw' in x]
        eindices = sindices[1:] + [len(raw_lines)]
        for start_idx, end_idx in zip(sindices, eindices):
            feat_dict = {}
            filename = raw_lines[start_idx].strip('[\n').rstrip().replace(\
                    'sw0', 'sw')
            if filename in done_files:
                print "already done: ", filename
                continue
            frames = raw_lines[start_idx+1:end_idx]
            list_feats = [f.strip().split()[:numc] for f in frames]
            floated_feats = [[float(x) for x in coef] for coef in list_feats]
            feat_dict[filename] = floated_feats
            full_name = os.path.join(output_dir, filename + '.pickle') 
            pickle.dump(feat_dict, open(full_name, 'w'))

if __name__ == '__main__':
    pa = argparse.ArgumentParser(
            description='Process kaldi features')
    pa.add_argument('--split', help='split')
    pa.add_argument('--in_dir', help='inpput directory', \
            default='/s0/ttmt001/speech_parsing/')
    pa.add_argument('--out_dir', help='output directory', \
            default='/s0/ttmt001/speech_parsing/')
    pa.add_argument('--feattype', help='feature type', \
            default='fbank_energy')
    args = pa.parse_args()
    process_feats(args)
    sys.exit(0)

