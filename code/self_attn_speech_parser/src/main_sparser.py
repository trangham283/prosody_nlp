import argparse
import itertools
import os.path
import time
import pickle

import torch
import torch.optim.lr_scheduler

import numpy as np
from glob import glob
import sys
import evaluate
import trees
import vocabulary
import nkutil
import parse_model
import chart_helper
tokens = parse_model

def torch_load(load_path):
    if parse_model.use_cuda:
        return torch.load(load_path)
    else:
        return torch.load(load_path, map_location=lambda storage, \
                location: storage)

def format_elapsed(start_time):
    elapsed_time = int(time.time() - start_time)
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    elapsed_string = "{}h{:02}m{:02}s".format(hours, minutes, seconds)
    if days > 0:
        elapsed_string = "{}d{}".format(days, elapsed_string)
    return elapsed_string

def make_hparams():
    return nkutil.HParams(
        max_len_train=0, # no length limit
        max_len_dev=0, # no length limit

        sentence_max_len=200,

        learning_rate=0.0008,
        learning_rate_warmup_steps=160,
        clip_grad_norm=0., #no clipping
        step_decay=True, # note that disabling step decay is not implemented
        step_decay_factor=0.5,
        step_decay_patience=5,
        max_consecutive_decays=3, # establishes a termination criterion

        partitioned=True,
        num_layers_position_only=0,

        num_layers=8,
        d_model=1024,
        num_heads=8,
        d_kv=64,
        d_ff=2048,
        d_label_hidden=250,
        d_tag_hidden=250,
        tag_loss_scale=5.0,

        attention_dropout=0.2,
        embedding_dropout=0.0,
        relu_dropout=0.1,
        residual_dropout=0.2,

        use_tags=False,
        use_words=False,
        use_pause=False,
        use_chars_lstm=False,
        use_chars_concat=False,
        use_elmo=False,
        use_bert=False,
        use_bert_only=False,
        use_glove_pretrained=False,
        use_glove_fisher=False,
        predict_tags=False,

        d_char_emb=32, # A larger value may be better for use_chars_lstm
        d_pause_emb=4,

        d_duration = 2,
        d_mfcc = 13,
        d_pitch = 3,
        d_fbank = 3,
        d_f0coefs = 12,
        fixed_word_length = 100,

        tag_emb_dropout=0.2,
        word_emb_dropout=0.4,
        morpho_emb_dropout=0.2,
        timing_dropout=0.0,
        char_lstm_input_dropout=0.2,
        elmo_dropout=0.5, # Note that this semi-stacks with morpho_emb_dropout!

        bert_model="bert-base-uncased",
        bert_do_lower_case=True,
        freeze=False, 
        )

def load_features(sent_ids, feat_dict, sp_off=False):
    if not feat_dict: 
        return None
    batch_features = {}
    for sent in sent_ids:
        features = {}
        for k in feat_dict.keys():
            if k == 'pause':
                features[k] = feat_dict[k][sent]['pause_aft']
                if sp_off:
                    # '3' category for 0 pause
                    features[k] = ['3' for _ in range(len(features[k]))]
            elif k in ['pitch', 'fbank', 'mfcc']:
                if 'frames' not in features.keys(): 
                    features['frames'] = feat_dict[k][sent]
                    if sp_off: 
                        features['frames'] = np.zeros(features['frames'].shape)
                else:
                    if sp_off:
                        features['frames'] = np.vstack([features['frames'], \
                                np.zeros(feat_dict[k][sent].shape)])
                    else:
                        features['frames'] = np.vstack([features['frames'], \
                                feat_dict[k][sent]])
            elif k in ['duration', 'f0coefs']:
                if 'scalars' not in features.keys():
                    features['scalars'] = feat_dict[k][sent]
                    if sp_off:
                        features['scalars'] = np.zeros(feat_dict[k][sent].shape)
                else:
                    if sp_off:
                        features['scalars'] = np.vstack([features['scalars'], \
                                np.zeros(feat_dict[k][sent].shape)])
                    else:
                        features['scalars'] = np.vstack([features['scalars'], \
                                feat_dict[k][sent]])
            else:
                # Partition feature is left
                features[k] = feat_dict[k][sent]
        batch_features[sent] = features
    return batch_features

def run_train(args, hparams):
    if args.numpy_seed is not None:
        print("Setting numpy random seed to {}...".format(args.numpy_seed))
        np.random.seed(args.numpy_seed)
        sys.stdout.flush()

    # Make sure that pytorch is actually being initialized randomly.
    # On my cluster I was getting highly correlated results from multiple
    # runs, but calling reset_parameters() changed that. A brief look at the
    # pytorch source code revealed that pytorch initializes its RNG by
    # calling std::random_device, which according to the C++ spec is allowed
    # to be deterministic.
    seed_from_numpy = np.random.randint(2147483648)
    print("Manual seed for pytorch:", seed_from_numpy)
    torch.manual_seed(seed_from_numpy)

    hparams.set_from_args(args)

    print("Loading training trees from {}...".format(args.train_path))
    train_treebank, train_sent_ids = trees.load_trees_with_idx(args.train_path,\
            args.train_sent_id_path)

    print("Processing pause features for training...")
    pause_path = os.path.join(args.feature_path, args.prefix + 'train_pause.pickle')
    with open(pause_path, 'rb') as f:
        pause_data = pickle.load(f, encoding='latin1')

    print("Processing trees for training...")
    wsj_sents = set([x for x in train_sent_ids if 'wsj' in x])
    if len(wsj_sents)> 0:
        assert args.speech_features is None
    to_keep = set(pause_data.keys())
    to_keep = to_keep.union(wsj_sents)
    train_parse = [tree.convert() for tree in train_treebank]
    # Removing sentences without speech info
    to_remove = set(train_sent_ids).difference(to_keep)
    to_remove = sorted([train_sent_ids.index(i) for i in to_remove])
    for x in to_remove[::-1]:
        train_parse.pop(x)
        train_sent_ids.pop(x)
    train_set = list(zip(train_sent_ids, train_parse))
    print("Loaded {:,} training examples.".format(len(train_set)))

    # Remove sentences without prosodic features in dev set
    print("Loading development trees from {}...".format(args.dev_path))
    dev_treebank, dev_sent_ids = trees.load_trees_with_idx(args.dev_path, \
            args.dev_sent_id_path)
    dev_pause_path = os.path.join(args.feature_path, args.prefix + \
            'dev_pause.pickle')
    with open(dev_pause_path, 'rb') as f:
        dev_pause_data = pickle.load(f, encoding='latin1')
    to_remove = set(dev_sent_ids).difference(set(dev_pause_data.keys()))
    to_remove = sorted([dev_sent_ids.index(i) for i in to_remove])
    for x in to_remove[::-1]:
        dev_treebank.pop(x)
        dev_sent_ids.pop(x)
    #if hparams.max_len_dev > 0:
    #    dev_treebank = [tree for tree in dev_treebank if \
    #            len(list(tree.leaves())) <= hparams.max_len_dev]
    print("Loaded {:,} development examples.".format(len(dev_treebank)))

    print("Constructing vocabularies...")
    sys.stdout.flush()

    tag_vocab = vocabulary.Vocabulary()
    tag_vocab.index(tokens.START)
    tag_vocab.index(tokens.STOP)
    tag_vocab.index(tokens.TAG_UNK)

    word_vocab = vocabulary.Vocabulary()
    word_vocab.index(tokens.START)
    word_vocab.index(tokens.STOP)
    word_vocab.index(tokens.UNK)

    pause_vocab = vocabulary.Vocabulary()
    pause_vocab.index(tokens.START)
    pause_vocab.index(tokens.STOP)

    label_vocab = vocabulary.Vocabulary()
    label_vocab.index(())

    char_set = set()

    for v in pause_data.values():
        pauses = v['pause_aft']
        for p in pauses:
            pause_vocab.index(str(p))

    for tree in train_parse:
        nodes = [tree]
        while nodes:
            node = nodes.pop()
            if isinstance(node, trees.InternalParseNode):
                label_vocab.index(node.label)
                nodes.extend(reversed(node.children))
            else:
                tag_vocab.index(node.tag)
                word_vocab.index(node.word)
                char_set |= set(node.word)

    char_vocab = vocabulary.Vocabulary()

    # If codepoints are small (e.g. Latin alphabet), index by codepoint directly
    highest_codepoint = max(ord(char) for char in char_set)
    if highest_codepoint < 512:
        if highest_codepoint < 256:
            highest_codepoint = 256
        else:
            highest_codepoint = 512

        # This also takes care of constants like tokens.CHAR_PAD
        for codepoint in range(highest_codepoint):
            char_index = char_vocab.index(chr(codepoint))
            assert char_index == codepoint
    else:
        char_vocab.index(tokens.CHAR_UNK)
        char_vocab.index(tokens.CHAR_START_SENTENCE)
        char_vocab.index(tokens.CHAR_START_WORD)
        char_vocab.index(tokens.CHAR_STOP_WORD)
        char_vocab.index(tokens.CHAR_STOP_SENTENCE)
        for char in sorted(char_set):
            char_vocab.index(char)

    tag_vocab.freeze()
    word_vocab.freeze()
    label_vocab.freeze()
    char_vocab.freeze()
    pause_vocab.freeze()

    def print_vocabulary(name, vocab):
        special = {tokens.START, tokens.STOP, tokens.UNK}
        print("{} ({:,}): {}".format(
            name, vocab.size,
            sorted(value for value in vocab.values if value in special) +
            sorted(value for value in vocab.values if value not in special)))

    if args.print_vocabs:
        print_vocabulary("Tag", tag_vocab)
        print_vocabulary("Word", word_vocab)
        print_vocabulary("Label", label_vocab)
        print_vocabulary("Pause", pause_vocab)

    feat_dict = {}
    speech_features = None
    if args.speech_features is not None:
        speech_features = args.speech_features.split(',')
        if 'pause' in speech_features:
            hparams.use_pause = True
        print("Loading speech features for training set...")
        for feat_type in speech_features:
            print("\t", feat_type)
            feat_path = os.path.join(args.feature_path, \
                    args.prefix + 'train_' + feat_type + '.pickle')
            with open(feat_path, 'rb') as f:
                feat_data = pickle.load(f, encoding='latin1')
            feat_dict[feat_type] = feat_data

    dev_feat_dict = {}
    if args.speech_features is not None:
        speech_features = args.speech_features.split(',')
        print("Loading speech features for dev set...")
        for feat_type in speech_features:
            print("\t", feat_type)
            feat_path = os.path.join(args.feature_path, \
                    args.prefix + 'dev_' + feat_type + '.pickle')
            with open(feat_path, 'rb') as f:
                feat_data = pickle.load(f, encoding='latin1')
            dev_feat_dict[feat_type] = feat_data

    print("Hyperparameters:")
    hparams.print()
    print("Initializing model...")
    sys.stdout.flush()

    load_path = args.load_path
    if load_path is not None:
        print("Loading parameters from ".format(load_path))
        info = torch_load(load_path)
        parser = parse_model.SpeechParser.from_spec(info['spec'], \
                info['state_dict'])
    else:
        parser = parse_model.SpeechParser(
            tag_vocab,
            word_vocab,
            label_vocab,
            char_vocab,
            pause_vocab,
            speech_features,
            hparams,
        )

    print("Initializing optimizer...")
    trainable_parameters = [param for param in parser.parameters() \
            if param.requires_grad]
    print(parser)
    for name, param in parser.named_parameters(): 
        print(name, param.data.shape, param.requires_grad)

    if args.optimizer == 'SGD':
        trainer = torch.optim.SGD(trainable_parameters, lr=0.05)
    else:
        trainer = torch.optim.Adam(trainable_parameters, \
                lr=hparams.learning_rate, \
                betas=(0.9, 0.98), eps=1e-9)
    if load_path is not None:
        trainer.load_state_dict(info['trainer'])

    def set_lr(new_lr):
        for param_group in trainer.param_groups:
            param_group['lr'] = new_lr

    assert hparams.step_decay, "Only step_decay schedule is supported"

    warmup_coeff = hparams.learning_rate / hparams.learning_rate_warmup_steps
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        trainer, 'max',
        factor=hparams.step_decay_factor,
        patience=hparams.step_decay_patience,
        verbose=True,
    )
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= hparams.learning_rate_warmup_steps:
            set_lr(iteration * warmup_coeff)

    clippable_parameters = trainable_parameters
    grad_clip_threshold = np.inf if hparams.clip_grad_norm == 0 \
            else hparams.clip_grad_norm

    print("Training...")
    total_processed = 0
    current_processed = 0
    check_every = len(train_parse) / args.checks_per_epoch
    best_dev_fscore = -np.inf
    best_dev_model_path = None
    best_dev_processed = 0

    start_time = time.time()
    
    def check_dev():
        nonlocal best_dev_fscore
        nonlocal best_dev_model_path
        nonlocal best_dev_processed

        dev_start_time = time.time()

        dev_predicted = []
        eval_batch_size = args.eval_batch_size
        for dev_start_index in range(0, len(dev_treebank), eval_batch_size):
            subbatch_trees = dev_treebank[dev_start_index:dev_start_index \
                    + eval_batch_size]
            subbatch_sent_ids = dev_sent_ids[dev_start_index:dev_start_index \
                    + eval_batch_size]
            subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in \
                    tree.leaves()] for tree in subbatch_trees]
            subbatch_features = load_features(subbatch_sent_ids, dev_feat_dict)

            predicted, _ = parser.parse_batch(subbatch_sentences, \
                    subbatch_sent_ids, subbatch_features)

            del _
            dev_predicted.extend([p.convert() for p in predicted])

        dev_fscore = evaluate.evalb(args.evalb_dir, dev_treebank, dev_predicted)

        print(
            "dev-fscore {} "
            "dev-elapsed {} "
            "total-elapsed {}".format(
                dev_fscore,
                format_elapsed(dev_start_time),
                format_elapsed(start_time),
            )
        )
        sys.stdout.flush()

        if dev_fscore.fscore > best_dev_fscore:
            if best_dev_model_path is not None:
                extensions = [".pt"]
                for ext in extensions:
                    path = best_dev_model_path + ext
                    if os.path.exists(path):
                        print("Removing previous model file {}...".format(path))
                        os.remove(path)

            best_dev_fscore = dev_fscore.fscore
            best_dev_model_path = "{}_dev={:.2f}".format(
                args.model_path_base, dev_fscore.fscore)
            best_dev_processed = total_processed
            print("Saving new best model to {}...".format(best_dev_model_path))
            torch.save({
                'spec': parser.spec,
                'state_dict': parser.state_dict(),
                'trainer' : trainer.state_dict(),
                }, best_dev_model_path + ".pt")
            sys.stdout.flush()

    for epoch in itertools.count(start=1):
        if args.epochs is not None and epoch > args.epochs:
            break

        np.random.shuffle(train_set)
        epoch_start_time = time.time()

        for start_index in range(0, len(train_set), args.batch_size):
            trainer.zero_grad()
            schedule_lr(total_processed // args.batch_size)

            batch_loss_value = 0.0
            batch_trees = [x[1] for x in \
                    train_set[start_index:start_index + args.batch_size]]
            batch_sent_ids = [x[0] for x in \
                    train_set[start_index:start_index + args.batch_size]]
            batch_sentences = [[(leaf.tag, leaf.word) for leaf \
                    in tree.leaves()] for tree in batch_trees]
            batch_num_tokens = sum(len(sentence) for sentence \
                    in batch_sentences)

            for subbatch_sentences, subbatch_trees, subbatch_sent_ids in \
                    parser.split_batch(batch_sentences, batch_trees, \
                    batch_sent_ids, args.subbatch_max_tokens):
                subbatch_features = load_features(subbatch_sent_ids, feat_dict) 
                
                _, loss = parser.parse_batch(subbatch_sentences, \
                        subbatch_sent_ids, subbatch_features, subbatch_trees)

                if hparams.predict_tags:
                    loss = loss[0]/len(batch_trees) + loss[1]/batch_num_tokens
                else:
                    loss = loss/len(batch_trees)
                loss_value = float(loss.data.cpu().numpy())
                batch_loss_value += loss_value
                if loss_value > 0:
                    loss.backward()
                del loss
                total_processed += len(subbatch_trees)
                current_processed += len(subbatch_trees)

            grad_norm = torch.nn.utils.clip_grad_norm_(clippable_parameters, \
                    grad_clip_threshold)

            trainer.step()

            print(
                "epoch {:,} "
                "batch {:,}/{:,} "
                "processed {:,} "
                "batch-loss {:.4f} "
                "grad-norm {:.4f} "
                "epoch-elapsed {} "
                "total-elapsed {}".format(
                    epoch,
                    start_index // args.batch_size + 1,
                    int(np.ceil(len(train_parse) / args.batch_size)),
                    total_processed,
                    batch_loss_value,
                    grad_norm,
                    format_elapsed(epoch_start_time),
                    format_elapsed(start_time),
                )
            )
            sys.stdout.flush()

            # DEBUG: 
            if args.debug and start_index > args.batch_size * 2:
                print("In debug mode, exiting")
                exit(0)

            if current_processed >= check_every:
                current_processed -= check_every
                check_dev()

        # adjust learning rate at the end of an epoch
        if (total_processed // args.batch_size + 1) > \
                hparams.learning_rate_warmup_steps:
            scheduler.step(best_dev_fscore)
            if (total_processed - best_dev_processed) > \
                    ((hparams.step_decay_patience + 1) \
                    * hparams.max_consecutive_decays * len(train_parse)):
                print("Terminating due to lack of improvement in dev fscore.")
                break

def run_test(args):
    print("Loading test trees from {}...".format(args.test_path))
    test_treebank, test_sent_ids = trees.load_trees_with_idx(args.test_path, \
            args.test_sent_id_path)
    
    if not args.new_set:
        test_pause_path = os.path.join(args.feature_path, args.test_prefix + \
            '_pause.pickle')
        with open(test_pause_path, 'rb') as f:
            test_pause_data = pickle.load(f, encoding='latin1')
        to_remove = set(test_sent_ids).difference(set(test_pause_data.keys()))
        to_remove = sorted([test_sent_ids.index(i) for i in to_remove])
        for x in to_remove[::-1]:
            test_treebank.pop(x)
            test_sent_ids.pop(x)

    print("Loaded {:,} test examples.".format(len(test_treebank)))

    print("Loading model from {}...".format(args.model_path_base))
    assert args.model_path_base.endswith(".pt"), "Only pytorch files supported"

    info = torch_load(args.model_path_base)
    print(info.keys())
    assert 'hparams' in info['spec'], "Older savefiles not supported"

    parser = parse_model.SpeechParser.from_spec(info['spec'], \
            info['state_dict'])
    parser.eval() # turn off dropout at evaluation time
    label_vocab = parser.label_vocab
    #print("{} ({:,}): {}".format("label", label_vocab.size, \
    #        sorted(value for value in label_vocab.values)))

    test_feat_dict = {}
    if info['spec']['speech_features'] is not None:
        speech_features = info['spec']['speech_features']
        print("Loading speech features for test set...")
        for feat_type in speech_features:
            print("\t", feat_type)
            feat_path = os.path.join(args.feature_path, \
                    args.test_prefix + '_' + feat_type + '.pickle')
            with open(feat_path, 'rb') as f:
                feat_data = pickle.load(f, encoding='latin1')
            test_feat_dict[feat_type] = feat_data

    print("Parsing test sentences...")
    start_time = time.time()

    test_predicted = []
    test_scores = []
    pscores = []
    gscores = []
    with torch.no_grad():
        for start_index in range(0, len(test_treebank), args.eval_batch_size):
            subbatch_treebank = test_treebank[start_index:start_index \
                    + args.eval_batch_size]
            subbatch_sent_ids = test_sent_ids[start_index:start_index \
                    + args.eval_batch_size]
            subbatch_sentences = [[(leaf.tag, leaf.word) for leaf in \
                    tree.leaves()] for tree in subbatch_treebank]
            subbatch_trees = [t.convert() for t in subbatch_treebank]
            subbatch_features = load_features(subbatch_sent_ids, test_feat_dict\
                    , args.sp_off)
            predicted, scores = parser.parse_batch(subbatch_sentences, \
                        subbatch_sent_ids, subbatch_features)
            if not args.get_scores:
                del scores
            else:
                charts = parser.parse_batch(subbatch_sentences, \
                        subbatch_sent_ids, subbatch_features, subbatch_trees, True)
                for i in range(len(charts)):
                    decoder_args = dict(sentence_len=len(subbatch_sentences[i]),\
                            label_scores_chart=charts[i],\
                            gold=subbatch_trees[i],\
                            label_vocab=parser.label_vocab, \
                            is_train=False, \
                            backoff=True)
                    p_score, _, _, _, _ = chart_helper.decode(False, **decoder_args)
                    g_score, _, _, _, _ = chart_helper.decode(True, **decoder_args)
                    pscores.append(p_score)
                    gscores.append(g_score)
                test_scores += scores
            test_predicted.extend([p.convert() for p in predicted])
    
    # DEBUG
    # print(test_scores)
    #print(test_score_offsets)

    with open(args.output_path, 'w') as output_file:
        for tree in test_predicted:
            output_file.write("{}\n".format(tree.linearize()))
    print("Output written to:", args.output_path)

    if args.get_scores:
        with open(args.output_path+'.scores', 'w') as output_file:
            for score1, score2, score3 in zip(test_scores, pscores, gscores):
                output_file.write("{}\t{}\t{}\n".format(score1, score2, score3))
        print("Output scores written to:", args.output_path+'.scores')


    if args.write_gold:
        with open(args.test_prefix + '_sent_ids.txt', 'w') as sid_file:
            for sent_id in test_sent_ids:
                sid_file.write("{}\n".format(sent_id))
        print("Sent ids written to:", args.test_prefix + '_sent_ids.txt')

        with open(args.test_prefix + '_gold.txt', 'w') as gold_file:
            for tree in test_treebank:
                gold_file.write("{}\n".format(tree.linearize()))
        print("Gold trees written to:", args.test_prefix + '_gold.txt')

    # The tree loader does some preprocessing to the trees (e.g. stripping TOP
    # symbols or SPMRL morphological features). We compare with the input file
    # directly to be extra careful about not corrupting the evaluation. We also
    # allow specifying a separate "raw" file for the gold trees: the inputs to
    # our parser have traces removed and may have predicted tags substituted,
    # and we may wish to compare against the raw gold trees to make sure we
    # haven't made a mistake. As far as we can tell all of these variations give
    # equivalent results.
    ref_gold_path = args.test_path
    if args.test_path_raw is not None:
        print("Comparing with raw trees from", args.test_path_raw)
        ref_gold_path = args.test_path_raw
    else:
        # Need this since I'm evaluating on subset
        ref_gold_path = None

    test_fscore = evaluate.evalb(args.evalb_dir, test_treebank, \
            test_predicted, ref_gold_path=ref_gold_path, is_train=False)

    print(
        "test-fscore {} "
        "test-elapsed {}".format(
            test_fscore,
            format_elapsed(start_time),
        )
    )

def run_viz(args):
    assert args.model_path_base.endswith(".pt"), "Only pytorch savefiles supported"

    print("Loading test trees from {}...".format(args.viz_path))
    viz_treebank, viz_sent_ids = trees.load_trees_with_idx(args.viz_path, \
            args.viz_sent_id_path)
    print("Loaded {:,} test examples.".format(len(viz_treebank)))

    print("Loading model from {}...".format(args.model_path_base))

    info = torch_load(args.model_path_base)

    assert 'hparams' in info['spec'], "Only self-attentive models are supported"
    parser = parse_model.SpeechParser.from_spec(info['spec'], \
            info['state_dict'])

    viz_feat_dict = {}
    if info['spec']['speech_features'] is not None:
        speech_features = info['spec']['speech_features']
        print("Loading speech features for test set...")
        for feat_type in speech_features:
            print("\t", feat_type)
            feat_path = os.path.join(args.feature_path, \
                    args.viz_prefix + '_' + feat_type + '.pickle')
            with open(feat_path, 'rb') as f:
                feat_data = pickle.load(f, encoding='latin1')
            viz_feat_dict[feat_type] = feat_data

    from viz import viz_attention

    stowed_values = {}
    orig_multihead_forward = parse_model.MultiHeadAttention.forward
    def wrapped_multihead_forward(self, inp, batch_idxs, **kwargs):
        res, attns = orig_multihead_forward(self, inp, batch_idxs, **kwargs)
        stowed_values['attns{}'.format(stowed_values["stack"])] = attns.cpu().data.numpy()
        stowed_values['stack'] += 1
        return res, attns

    parse_model.MultiHeadAttention.forward = wrapped_multihead_forward

    # Select the sentences we will actually be visualizing
    max_len_viz = 40
    if max_len_viz > 0:
        viz_treebank = [tree for tree in viz_treebank if len(list(tree.leaves())) <= max_len_viz]
    #viz_treebank = viz_treebank[:1]

    print("Parsing viz sentences...")

    viz_data = []

    for start_index in range(0, len(viz_treebank), args.eval_batch_size):
        subbatch_trees = viz_treebank[start_index:start_index + \
                args.eval_batch_size]
        subbatch_sent_ids = viz_sent_ids[start_index:start_index + \
                args.eval_batch_size]
        subbatch_sentences = [[(leaf.tag, leaf.word) for \
                leaf in tree.leaves()] for tree in subbatch_trees]
        subbatch_features = load_features(subbatch_sent_ids, viz_feat_dict)
        stowed_values = dict(stack=0)
        predicted, _ = parser.parse_batch(subbatch_sentences, \
                    subbatch_sent_ids, subbatch_features)
        del _
        predicted = [p.convert() for p in predicted]
        stowed_values['predicted'] = predicted

        for snum, sentence in enumerate(subbatch_sentences):
            sentence_words = [tokens.START] + [x[1] for x in sentence] + [tokens.STOP]

            for stacknum in range(stowed_values['stack']):
                attns_padded = stowed_values['attns{}'.format(stacknum)]
                attns = attns_padded[snum::len(subbatch_sentences), :len(sentence_words), :len(sentence_words)]
                dat = viz_attention(sentence_words, attns)
                viz_data.append(dat)

    outf = open(args.viz_out, 'wb')
    pickle.dump(viz_data, outf)
    outf.close()

def run_parse(args):
    # NOTE: This only supports text-only version for now
    # With new dataset that contains audio features the prosodic feature 
    # extraction process might need to be different 

    print("Loading model from {}...".format(args.model_path_base))
    assert args.model_path_base.endswith(".pt"), "Only pytorch files supported"

    info = torch_load(args.model_path_base)
    assert 'hparams' in info['spec'], "Older savefiles not supported"
    parser = parse_model.SpeechParser.from_spec(info['spec'], \
            info['state_dict'])

    print("Parsing sentences...")
    
    all_files = glob(args.input_path + "/*.txt")
    for this_file in all_files:
        basename = os.path.basename(this_file)[:-4]
        print(basename)
        with open(this_file) as input_file:
            sentences = input_file.readlines()
        sentences = [sentence.split() for sentence in sentences]

        # Tags are not available when parsing from raw text, so use a dummy tag
        if 'UNK' in parser.tag_vocab.indices:
            dummy_tag = 'UNK'
        else:
            dummy_tag = parser.tag_vocab.value(0)

        start_time = time.time()

        all_predicted = []
        for start_index in range(0, len(sentences), args.eval_batch_size):
            subbatch_sentences = sentences[start_index:start_index+args.eval_batch_size]

            subbatch_sentences = [[(dummy_tag, word) for word in sentence] \
                    for sentence in subbatch_sentences]
            # dummy subbatch sent_ids and features
            subbatch_sent_ids = [0]*len(subbatch_sentences) 
            subbatch_features = {} 
            predicted, _ = parser.parse_batch(subbatch_sentences, \
                    subbatch_sent_ids, subbatch_features)
            del _
            if args.output_path == '-':
                for p in predicted:
                    print(p.convert().linearize())
            else:
                all_predicted.extend([p.convert() for p in predicted])

        with open(args.output_path + "/" + basename + ".parse", 'w') as output_file:
            for tree in all_predicted:
                to_write = '(ROOT ' + tree.linearize() + ')'
                output_file.write("{}\n".format(to_write))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    hparams = make_hparams()
    subparser = subparsers.add_parser("train")
    subparser.set_defaults(callback=lambda args: run_train(args, hparams))
    hparams.populate_arguments(subparser)
    subparser.add_argument("--numpy-seed", type=int)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--feature-path", \
      default="/Users/trangtran/Misc/data/swbd_features")
    subparser.add_argument("--train-path", \
      default="/Users/trangtran/Misc/data/swbd_trees/swbd_train2.txt")
    subparser.add_argument("--dev-path", \
      default="/Users/trangtran/Misc/data/swbd_trees/swbd_dev.txt")
    subparser.add_argument("--train-sent-id-path", \
      default="/Users/trangtran/Misc/data/swbd_trees/train2_sent_ids.txt")
    subparser.add_argument("--dev-sent-id-path", \
      default="/Users/trangtran/Misc/data/swbd_trees/dev_sent_ids.txt")
    subparser.add_argument("--load-path", type=str, default=None)
    subparser.add_argument("--prefix", type=str, default='')
    subparser.add_argument("--optimizer", type=str, default='adam')
    subparser.add_argument("--speech-features", type=str, default=None)
    subparser.add_argument("--batch-size", type=int, default=250)
    subparser.add_argument("--subbatch-max-tokens", type=int, default=2000)
    subparser.add_argument("--eval-batch-size", type=int, default=100)
    subparser.add_argument("--epochs", type=int)
    subparser.add_argument("--checks-per-epoch", type=int, default=4)
    subparser.add_argument("--print-vocabs", action="store_true")
    subparser.add_argument("--debug", action="store_true", default=False)

    subparser = subparsers.add_parser("test")
    subparser.set_defaults(callback=run_test)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--feature-path", \
        default="/Users/trangtran/Misc/data/swbd_features")
    subparser.add_argument("--test-path", \
        default="/Users/trangtran/Misc/data/swbd_trees/swbd_test.txt")
    subparser.add_argument("--test-sent-id-path", \
        default="/Users/trangtran/Misc/data/swbd_trees/test_sent_ids.txt")
    subparser.add_argument("--speech-features", type=str, default=None)
    subparser.add_argument("--test-prefix", type=str, default='test')
    subparser.add_argument("--output-path", type=str, \
        default="/Users/trangtran/Misc/data/parser_out/test_predicted.txt")
    subparser.add_argument("--test-path-raw", type=str, default=None)
    subparser.add_argument("--eval-batch-size", type=int, default=100)
    subparser.add_argument("--sp-off", action="store_true", default=False,\
            help="Turn off speech features to check effect")
    subparser.add_argument("--get-scores", action="store_true", default=False,\
            help="Also return tree score if set")
    subparser.add_argument("--write-gold", action="store_true", default=False,\
            help="Write gold trees for reference")
    subparser.add_argument("--new-set", action="store_true", default=False, \
            help="Flag for a new data set (which doesn't have speech features")

    subparser = subparsers.add_parser("viz")
    subparser.set_defaults(callback=run_viz)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--evalb-dir", default="EVALB/")
    subparser.add_argument("--feature-path", \
        default="/g/ssli/data/CTS-English/swbd_align/swbd_features")
    subparser.add_argument("--viz-path", \
        default="L1_sample.txt")
    subparser.add_argument("--viz-sent-id-path", \
        default="L1_sample_sent_ids.txt")
    subparser.add_argument("--viz-prefix", type=str, default='L1')
    subparser.add_argument("--viz-out", type=str, default='L1_sample.pickle')
    subparser.add_argument("--eval-batch-size", type=int, default=1)

    subparser = subparsers.add_parser("parse")
    subparser.set_defaults(callback=run_parse)
    subparser.add_argument("--model-path-base", required=True)
    subparser.add_argument("--input-path", type=str, required=True)
    subparser.add_argument("--output-path", type=str, default="-")
    subparser.add_argument("--eval-batch-size", type=int, default=100)

    args = parser.parse_args()
    args.callback(args)

# %%
if __name__ == "__main__":
    main()
