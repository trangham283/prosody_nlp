i = 0
for snum, sentence in enumerate(sentences):
    for wordnum, (tag, word) in \
        enumerate([(START, START)] + sentence + [(STOP, STOP)]):
        j = 0
        char_idxs_encoder[i, j] = parser.char_vocab.index(
                CHAR_START_WORD)
        j += 1
        if word in (START, STOP):
            char_idxs_encoder[i, j:j+3] = parser.char_vocab.index(
                CHAR_START_SENTENCE if (word == START) \
                        else CHAR_STOP_SENTENCE)
            j += 3
        else:
            for char in word:
                char_idxs_encoder[i, j] = \
                        parser.char_vocab.index_or_unk(char, CHAR_UNK)
                j += 1
        char_idxs_encoder[i, j] = \
                parser.char_vocab.index(CHAR_STOP_WORD)
        word_lens_encoder[i] = j + 1
        i += 1
assert i == packed_len


feature_path = "/Users/trangtran/Misc/data/swbd_features"

for feat_type in speech_features:
    print("\t", feat_type)
    feat_path = os.path.join(feature_path, \
            'dev_' + feat_type + '.pickle')
    with open(feat_path, 'rb') as f:
        feat_data = pickle.load(f, encoding='latin1')
    feat_dict[feat_type] = feat_data

def split_batch(sentences, golds, sent_ids, subbatch_max_tokens=3000):
    lens = [len(sentence) + 2 for sentence in sentences]
    lens = np.asarray(lens, dtype=int)
    lens_argsort = np.argsort(lens).tolist()
    num_subbatches = 0
    subbatch_size = 1
    while lens_argsort:
        if (subbatch_size == len(lens_argsort)) or (subbatch_size * lens[lens_argsort[subbatch_size]] > subbatch_max_tokens):
            yield [sentences[i] for i in lens_argsort[:subbatch_size]], [golds[i] for i in lens_argsort[:subbatch_size]], [sent_ids[i] for i in lens_argsort[:subbatch_size]]
            lens_argsort = lens_argsort[subbatch_size:]
            num_subbatches += 1
            subbatch_size = 1
        else:
            subbatch_size += 1

for snum, sentence in enumerate(sentences):
    for (tag, word) in [(START, START)] + sentence + [(STOP, STOP)]:
        tag_idxs[i] = 0 if not parser.use_tags \
                else tag_vocab.index_or_unk(tag, TAG_UNK)
        if word not in (START, STOP):
            count = word_vocab.count(word)
            if not count or \
                    (is_train and np.random.rand() < 1 / (1 + count)):
                word = UNK
        word_idxs[i] = word_vocab.index(word)
        batch_idxs[i] = snum
        i += 1
assert i == packed_len

def prep_features(sent_ids, sfeatures, pause_vocab):
    pause_features = []
    frame_features = []
    scalar_features = []
    for sent in sent_ids:
        sent_features = sfeatures[sent]
        if 'pause' in sent_features.keys():
            sent_pauses = [START] + [str(i) for i in sent_features['pause']] \
                    + [STOP]
            sent_pauses = [pause_vocab.index(x) for x in sent_pauses]
            pause_features += sent_pauses
        if 'scalars' in sent_features.keys():
            sent_scalars = sent_features['scalars']
            feat_dim = sent_scalars.shape[0]
            sent_scalar_feat = np.hstack([np.zeros((feat_dim, 1)), \
                    sent_scalars, \
                    np.zeros((feat_dim, 1))])
            scalar_features.append(sent_scalar_feat)
     return pause_features, frame_features, scalar_features

def process_sent_frames(sent_partition, sent_frames):
    feat_dim = sent_frames.shape[0]
    speech_frames = []
    for frame_idx in sent_partition:
        center_frame = int((frame_idx[0] + frame_idx[1])/2)
        start_idx = center_frame - int(fixed_word_length/2)
        end_idx = center_frame + int(fixed_word_length/2)
        raw_word_frames = sent_frames[:, frame_idx[0]:frame_idx[1]]
        raw_count = raw_word_frames.shape[1]
        if raw_count > fixed_word_length:
            this_word_frames = sent_frames[:, frame_idx[0]:frame_idx[1]]
            extra_ratio = int(raw_count/fixed_word_length)
            if extra_ratio < 2:  # delete things in the middle
                mask = np.ones(raw_count, dtype=bool)
                num_extra = raw_count - fixed_word_length
                not_include = range(center_frame-num_extra,
                                    center_frame+num_extra)[::2]
                not_include = [x-frame_idx[0] for x in not_include]
                mask[not_include] = False
            else:  # too big, just sample
                mask = np.zeros(raw_count, dtype=bool)
                include = range(frame_idx[0], frame_idx[1])[::extra_ratio]
                include = [x-frame_idx[0] for x in include]
                if len(include) > fixed_word_length:
                    num_current = len(include)
                    sub_extra = num_current - fixed_word_length
                    num_start = int((num_current - sub_extra)/2)
                    not_include = include[num_start:num_start+sub_extra]
                    for ni in not_include:
                        include.remove(ni)
                mask[include] = True
            this_word_frames = this_word_frames[:, mask]
        else:  # not enough frames, choose frames extending from center
            this_word_frames = sent_frames[:, max(0, start_idx):end_idx]
            if this_word_frames.shape[1] == 0:
                # make 0 if no frame info
                this_word_frames = np.zeros((feat_dim, fixed_word_length))
            if start_idx < 0 and this_word_frames.shape[1] < fixed_word_length:
                this_word_frames = np.hstack(
                    [np.zeros((feat_dim, -start_idx)), this_word_frames])
            if this_word_frames.shape[1] < fixed_word_length:
                num_more = fixed_word_length-this_word_frames.shape[1]
                this_word_frames = np.hstack(
                    [this_word_frames, np.zeros((feat_dim, num_more))])
        speech_frames.append(this_word_frames)
    sent_frame_features = [np.zeros((feat_dim, fixed_word_length))] \
        + speech_frames + [np.zeros((feat_dim, fixed_word_length))] 
    return sent_frame_features

