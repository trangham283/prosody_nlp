import argparse
import os
import pandas as pd
from glob import glob

data_dir = "/workspace/trang_tran/data/lls/prompts"
out_dir = "/workspace/trang_tran/data/swbd_trees"

files = glob(data_dir + "/*.tsv")

L1_sent_file = open(os.path.join(out_dir, 'L1_sent_ids.txt'), 'w')
L1_parse_file = open(os.path.join(out_dir, 'L1_parses.txt'), 'w')
L2_sent_file = open(os.path.join(out_dir, 'L2_sent_ids.txt'), 'w')
L2_parse_file = open(os.path.join(out_dir, 'L2_parses.txt'), 'w')

missing = ['S14_0028_L1', 'S15_0026_L1', 'S15_0028_L1', 'S31_0063_L1', 
        'S31_0103_L1', 'S43_0041_L1', 'S27_0078_L1']

for data_file in files:
    df = pd.read_csv(data_file, sep='\t')
    for i, row in df.iterrows():
        parse = row.parse_norm
        parse = parse.replace("ROOT", "TOP")
        sent_name = row.wav_id + '_L2'
        L2_sent_file.write(sent_name + "\n")
        L2_parse_file.write(parse + "\n")

        sent_name = row.wav_id + '_L1'
        if sent_name in missing:
            print sent_name
            continue
            
        L1_sent_file.write(sent_name + "\n")
        L1_parse_file.write(parse + "\n")
            
L1_sent_file.close()
L2_sent_file.close()
L1_parse_file.close()
L2_parse_file.close()

