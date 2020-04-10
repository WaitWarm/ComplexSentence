import subprocess

if __name__ == "__main__":
    # -train_doc corpus.doc -valid_src dev2010.zh-en.zh -valid_tgt dev2010.zh-en.en -valid_doc doc_dev_file -save_data out_file

    # a = subprocess.call(
    #     ['python3', 'preprocess.py', '-train_src',
    #      'corpus.tc.zh', '-train_tgt',
    #      'corpus.tc.en', '-train_doc',
    #      'corpus.doc', '-valid_src', 'IWSLT15.TED.dev2010.tc.zh', '-valid_tgt',
    #      'IWSLT15.TED.dev2010.tc.en', '-valid_doc',
    #      'dev2010.zh-en.doc', '-save_data', ''])
    #
    # b = subprocess.call(
    #     ['/home/xdjf/anaconda3/envs/torch1.4_10.2/bin/python3', 'train.py', '-data', 'IWSLT15.TED', '-save_model', 'HAN_joint_model', '-encoder_type',
    #      'transformer',
    #      '-decoder_type', 'transformer', '-enc_layers', '6', '-dec_layers', '6', '-label_smoothing',
    #      '0.1', '-src_word_vec_size', '512', '-tgt_word_vec_size', '512', '-rnn_size', '512', '-dropout', '0.1',
    #      '-batch_size', '1024', '-start_decay_at', '2', '-report_every', '500', '-epochs', '15', '-gpuid', '0',
    #      '-max_generator_batches', '32', '-batch_type', 'tokens', '-normalization', 'tokens', '-accum_count', '4',
    #      '-optim', 'adam', '-adam_beta2', '0.98', '-decay_method', 'noam', '-warmup_steps', '8000', '-learning_rate',
    #      '2', '-max_grad_norm', '0', '-param_init', '0', '-train_part', 'all', '-context_type', 'HAN_enc',
    #      '-context_size', '3', '-train_from', '', '-position_encoding', '-train_from',
    #      'sentence_level_model_acc_42.97_ppl_25.18_e13.pt'])
    # b_1 = subprocess.call(
    #     ['python3', 'train.py', '-data', 'IWSLT15.TED', '-save_model', 'sentence_level_model',
    #      '-encoder_type', 'transformer', '-decoder_type', 'transformer', '-enc_layers', '6', '-dec_layers', '6',
    #      '-label_smoothing', '0.1', '-src_word_vec_size', '512', '-tgt_word_vec_size', '512', '-rnn_size', '512',
    #      '-dropout', '0.1',
    #      '-batch_size', '1024', '-start_decay_at', '20', '-report_every', '500', '-epochs', '500', '-gpuid', '0',
    #      '-max_generator_batches',
    #      '16', '-batch_typ', 'tokens', '-normalization', 'tokens', '-accum_count', '4', '-optim', 'adam', '-adam_beta2',
    #      '0.998', '-decay_method', 'noam', '-warmup_steps', '4000', '-learning_rate', '2', '-max_grad_norm', '0',
    #      '-param_init',
    #      '0', '-param_init_glorot', '-train_part', 'sentences', '-position_encoding'])
    # python
    # translate.py - model[model] - src[test_source_file] - doc[test_doc_file]
    # -output[out_file] - translate_part
    # all - batch_size
    # 1000 - gpu
    # 0
    c = subprocess.call(['python3', 'translate.py', '-model', 'HAN_joint_model_acc_42.00_ppl_29.42_e1.pt', '-src',
                         'IWSLT15.TED.tst2010.tc.zh', '-tgt', 'IWSLT15.TED.tst2010.tc.en', '-doc',
                         'IWSLT15.TED.tst2010.zh-en.doc', '-output', 'test_out', '-translate_part', 'all',
                         '-batch_size', '1000', '-gpu', '0', '-report_bleu'])
    # d = subprocess.call(
    #     ['python3', 'preprocess.py',
    #      '-train_src',
    #      '../preprocess_TED_zh-en/corpus.tok.zh', '-train_tgt',
    #      '../preprocess_TED_zh-en/corpus.tok.en', '-train_doc',
    #      '../preprocess_TED_zh-en/corpus.doc', '-valid_src', '../preprocess_TED_zh-en/IWSLT15.TED.dev2010.tc.zh',
    #      '-valid_tgt', '../preprocess_TED_zh-en/IWSLT15.TED.dev2010.tc.en','-valid_doc','../preprocess_TED_zh-en/IWSLT15.TED.dev2010.zh-en.doc',
    #      '-save_data', '../preprocess_TED_zh-en/IWSLT15.TED'])
