#! /bin/bash
# Exit on error: https://stackoverflow.com/a/1379904/911441
set -e

train_set=train_si284
dev_sets=cv_dev93
dest_dir=
nj_train=16
nj_decode=8
dataset=wsj_8k

. ${KALDI_ROOT}/egs/wsj/s5/utils/parse_options.sh

echo $train_set
echo $dev_sets
echo $dest_dir

cd ${dest_dir}

green='\033[0;32m'
NC='\033[0m' # No Color
trap 'echo -e "${green}$ $BASH_COMMAND ${NC}"' DEBUG

export train_cmd=run.pl
export decode_cmd="run.pl --mem 2G"

[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C

for x in ${train_set} $dev_sets; do
    utils/fix_data_dir.sh data/$dataset/$x
done


################################################################################
# Extract MFCC features
#############################################################################

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
export mfccdir=mfcc
for x in ${train_set} $dev_sets; do
    steps/make_mfcc.sh --nj 20 --cmd "$train_cmd" \
                        data/$dataset/$x exp/$dataset/make_mfcc/$x $mfccdir
    steps/compute_cmvn_stats.sh data/$dataset/$x exp/$dataset/make_mfcc/$x $mfccdir
    utils/fix_data_dir.sh data/$dataset/$x
done

# Get the si-84 subset
utils/subset_data_dir.sh --first data/$dataset/$train_set 7138 data/$dataset/train_si84 || exit 1

# Now make subset with the shortest 2k utterances from si-84.
utils/subset_data_dir.sh --shortest data/$dataset/train_si84 2000 data/$dataset/train_si84_2kshort || exit 1;

# Now make subset with half of the data from si-84.
utils/subset_data_dir.sh data/$dataset/train_si84 3500 data/$dataset/train_si84_half || exit 1;


################################################################################
# Train monophone model on the 2k shortest utterances from the si-84 subset
###############################################################################
# Starting basic training on MFCC features
steps/train_mono.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
      data/$dataset/train_si84_2kshort data/lang_nosp exp/$dataset/mono0a || exit 1;

steps/align_si.sh --boost-silence 1.25 --nj 10 --cmd "$train_cmd" \
      data/$dataset/train_si84_half data/lang_nosp exp/$dataset/mono0a exp/$dataset/mono0a_ali || exit 1;
                
utils/mkgraph.sh data/lang_nosp_test_tgpr exp/$dataset/mono0a exp/$dataset/mono0a/graph_nosp_tgpr || exit 1;


################################################################################
# Create first triphone recognizer.
################################################################################
steps/train_deltas.sh --boost-silence 1.25 --cmd "$train_cmd" 2000 10000 \
      data/$dataset/train_si84_half data/lang_nosp exp/$dataset/mono0a_ali exp/$dataset/tri1 || exit 1;


steps/align_si.sh --nj 10 --cmd "$train_cmd" \
      data/$dataset/train_si84 data/lang_nosp exp/$dataset/tri1 exp/$dataset/tri1_ali_si84 || exit 1;

utils/mkgraph.sh data/lang_nosp_test_tgpr exp/$dataset/tri1 exp/$dataset/tri1/graph_nosp_tgpr || exit 1;


################################################################################
# Train LDA-MLLT model (tri2b) with all alignment steps
################################################################################

steps/train_lda_mllt.sh --cmd "$train_cmd" \
      --splice-opts "--left-context=3 --right-context=3" 2500 15000 \
      data/$dataset/train_si84 data/lang_nosp exp/$dataset/tri1_ali_si84 exp/$dataset/tri2b || exit 1;


steps/align_si.sh  --nj 10 --cmd "$train_cmd" \
      data/$dataset/train_si284 data/lang_nosp exp/$dataset/tri2b exp/$dataset/tri2b_ali_si284  || exit 1;

utils/mkgraph.sh data/lang_nosp_test_tgpr exp/$dataset/tri2b exp/$dataset/tri2b/graph_nosp_tgpr || exit 1;


################################################################################
# Train sat model (tri3b) with all alignment steps
################################################################################

steps/train_sat.sh --cmd "$train_cmd" 4200 40000 \
      data/$dataset/train_si284 data/lang_nosp exp/$dataset/tri2b_ali_si284 exp/$dataset/tri3b || exit 1;


utils/mkgraph.sh data/lang_nosp_test_tgpr exp/$dataset/tri3b exp/$dataset/tri3b/graph_nosp_tgpr || exit 1;

################################################################################
# Estimate pronunciation and silence probabilities.
################################################################################
  # Silprob for normal lexicon.
steps/get_prons.sh --cmd "$train_cmd" \
	data/$dataset/train_si284 data/lang_nosp exp/$dataset/tri3b || exit 1;
utils/dict_dir_add_pronprobs.sh --max-normalize true \
	data/local/dict_nosp \
	exp/$dataset/tri3b/pron_counts_nowb.txt exp/$dataset/tri3b/sil_counts_nowb.txt \
	exp/$dataset/tri3b/pron_bigram_counts_nowb.txt data/local/dict || exit 1

utils/prepare_lang.sh data/local/dict \
"<SPOKEN_NOISE>" data/local/lang_tmp data/lang || exit 1;

for lm_suffix in bg bg_5k tg tg_5k tgpr tgpr_5k; do
	mkdir -p data/lang_test_${lm_suffix}
	cp -r data/lang/* data/lang_test_${lm_suffix}/ || exit 1;
	rm -rf data/lang_test_${lm_suffix}/tmp
	cp data/lang_nosp_test_${lm_suffix}/G.* data/lang_test_${lm_suffix}/
done

# Silprob for larger ("bd") lexicon.
utils/dict_dir_add_pronprobs.sh --max-normalize true \
	data/local/dict_nosp_larger \
	exp/$dataset/tri3b/pron_counts_nowb.txt exp/$dataset/tri3b/sil_counts_nowb.txt \
	exp/$dataset/tri3b/pron_bigram_counts_nowb.txt data/local/dict_larger || exit 1

utils/prepare_lang.sh data/local/dict_larger \
	"<SPOKEN_NOISE>" data/local/lang_tmp_larger data/lang_bd || exit 1;

for lm_suffix in tgpr tgconst tg fgpr fgconst fg; do
	mkdir -p data/lang_test_bd_${lm_suffix}
	cp -r data/lang_bd/* data/lang_test_bd_${lm_suffix}/ || exit 1;
	rm -rf data/lang_test_bd_${lm_suffix}/tmp
	cp data/lang_nosp_test_bd_${lm_suffix}/G.* data/lang_test_bd_${lm_suffix}/
done

################################################################################
# From 3b system, now using data/lang as the lang directory (we have now added
# pronunciation and silence probabilities), train another SAT system (tri4b).
# Features: MFCC
################################################################################
steps/train_sat.sh  --cmd "$train_cmd" 4200 40000 \
      data/$dataset/train_si284 data/lang exp/$dataset/tri3b exp/tri4b || exit 1;

steps/align_fmllr.sh --nj 30 --cmd "$train_cmd" \
  data/$dataset/train_si284 data/lang exp/tri4b exp/$dataset/tri4b_ali_si284 || exit 1;

steps/align_fmllr.sh --nj 10 --cmd "$train_cmd" \
  data/$dataset/test_dev93 data/lang exp/$dataset/tri4b exp/tri4b_ali_test_dev93 || exit 1;

################################################################################
