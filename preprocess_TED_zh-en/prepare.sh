#!/bin/bash
moses_scripts=/root/mosesdecoder/mosesdecoder-master/scripts

zh_segment_home=/user/bin/python3
#kpu_preproc_dir=/fs/zisa0/bhaddow/code/preprocess/build/bin

max_len=200

export PYTHONPATH=$zh_segment_home

src=en
tgt=zh
pair=$src-$tgt


# Tokenise the English part
cat corpus.$tgt | \
$moses_scripts/tokenizer/normalize-punctuation.perl -l $tgt | \
$moses_scripts/tokenizer/tokenizer.perl -a -l $tgt  \
> corpus.tok.$tgt

#Segment the Chinese part
python3 -m jieba -d ' ' < corpus.$src > corpus.tok.$src

#
###
#### Clean
#$moses_scripts/training/clean-corpus-n.perl corpus.tok $src $tgt corpus.clean 1 $max_len corpus.retained
###
#

#### Train truecaser and truecase
$moses_scripts/recaser/train-truecaser.perl -model truecase-model.$tgt -corpus corpus.tok.$tgt
$moses_scripts/recaser/truecase.perl < corpus.tok.$tgt > corpus.tc.$tgt -model truecase-model.$tgt

ln -s corpus.tok.$src  corpus.tc.$src
#
#  
# dev sets
for devset in dev2010 tst2010 tst2011 tst2012 tst2013; do
  for lang  in $src $tgt; do
    if [ $lang = $tgt ]; then
      side="src"
      $moses_scripts/tokenizer/normalize-punctuation.perl -l $lang < IWSLT15.TED.$devset.$src-$tgt.$lang | \
      $moses_scripts/tokenizer/tokenizer.perl -a -l $lang |  \
      $moses_scripts/recaser/truecase.perl -model truecase-model.$lang \
      > IWSLT15.TED.$devset.tc.$lang
    else
      side="ref"
      python3 -m jieba -d ' '  < IWSLT15.TED.$devset.$src-$tgt.$lang > IWSLT15.TED.$devset.tc.$lang
    fi
    
  done

done

python3 /root/HAN_NMT-master/full_source/preprocess.py -train_src ../preprocess_TED_zh-en/corpus.tc.zh -train_tgt ../preprocess_TED_zh-en/corpus.tc.en -train_doc ../preprocess_TED_zh-en/corpus.doc -valid_src ../preprocess_TED_zh-en/IWSLT15.TED.dev2010.tc.zh -valid_tgt ../preprocess_TED_zh-en/IWSLT15.TED.dev2010.tc.en -valid_doc ../preprocess_TED_zh-en/IWSLT15.TED.dev2010.zh-en.doc -save_data ../preprocess_TED_zh-en/IWSLT15.TED -src_vocab_size 30000 -tgt_vocab_size 30000 -src_seq_length 80 -tgt_seq_length 80

