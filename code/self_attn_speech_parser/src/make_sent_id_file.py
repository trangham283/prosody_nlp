import argparse
import os
import pandas as pd

data_dir = "/Users/trangtran/Misc/data/swbd_trees"

def convert_name(speaker, sent_id):
    fnum, snum = sent_id.split('~')
    return '_'.join([fnum, speaker, snum])

def make_sent_ids(split):
    data_file = os.path.join(data_dir, split + '_mrg.tsv')
    df = pd.read_csv(data_file, sep='\t')
    df['sent_name'] = df.apply(lambda x: \
            convert_name(x.speaker, x.sent_id), axis=1)
    sent_file = open(os.path.join(data_dir, split + '_sent_ids.txt'), 'w')
    for s in df.sent_name.values:
        sent_file.write(s + '\n')
    sent_file.close()

pa = argparse.ArgumentParser(description='make sent id files')
pa.add_argument('--split', type=str, default='dev')
pa.add_argument('--foo', type=str, default=None)
args = pa.parse_args()
split = args.split
foo = args.foo
print(foo, foo is None)
#make_sent_ids(split)

