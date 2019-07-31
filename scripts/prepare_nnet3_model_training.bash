#! /bin/bash
# Exit on error: https://stackoverflow.com/a/1379904/911441
set -e

dest_dir=
nj=16
dataset=sms
train_set=train_si284
cv_sets=cv_dev93
train_cmd=run.pl
gmm=tri4b
gmm_data_type=wsj_8k
ali_data_type=sms_early
stage=5

num_threads_ubm=32
nnet3_affix=       # affix for exp dirs, e.g. it was _cleaned in tedlium.



. ./cmd.sh
. ./path.sh
. ${KALDI_ROOT}/egs/wsj/s5/utils/parse_options.sh

cd ${dest_dir}

green='\033[0;32m'
NC='\033[0m' # No Color
trap 'echo -e "${green}$ $BASH_COMMAND ${NC}"' DEBUG


################################################################################
# Extract MFCC features
#############################################################################

# Now make MFCC features.
# mfccdir should be some place with a largish disk where you
# want to store MFCC features.
export mfccdir=mfcc
if [ $stage -le 6 ]; then
    for x in ${train_set} $cv_sets ; do
        steps/make_mfcc.sh --nj $nj --cmd "$train_cmd" \
                            data/$dataset/$x exp/$dataset/make_mfcc/$x $mfccdir
        steps/compute_cmvn_stats.sh data/$dataset/$x exp/$dataset/make_mfcc/$x $mfccdir
        utils/fix_data_dir.sh data/$dataset/$x
    done
fi
################################################################################
# cleanup data
#############################################################################
# ToDo: should we use kaldi clean up?
#steps/cleanup/clean_and_segment_data.sh --nj 64 --cmd run.pl \
#    --segmentation-opts "--min-segment-length 0.3 --min-new-segment-length 0.6" \
#    data/${dataset}/$train_set data/lang exp/$gmm exp/{dataset}/tri4b_cleaned \
#    data/${dataset}/${train_set}_cleaned

################################################################################
# Estimate ivectors
#############################################################################
# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 11" if you have already
# run those things.
local_sms/run_ivector_common.sh \
            --stage $stage --nj $nj \
            --test_sets $cv_sets \
            --train-set $train_set --dataset $dataset \
            --gmm $gmm --gmm_data_type $gmm_data_type\
            --ali_data_type $ali_data_type \
            --num-threads-ubm $num_threads_ubm \
            --nnet3-affix "$nnet3_affix" || exit 1;


                                  
gmm_dir=exp/$gmm_data_type/${gmm}
ali_dir=exp/$ali_data_type/${gmm}_ali_${train_set}_sp
lat_dir=exp/$dataset/chain${nnet3_affix}/${gmm}_${train_set}_sp_lats
dir=exp/$dataset/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=data/$dataset/${train_set}_sp_hires
train_ivector_dir=exp/$dataset/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
lores_train_data_dir=data/$ali_data_type/${train_set}_sp


################################################################################
# get tree, lats etc
#############################################################################
# note: you don't necessarily have to change the treedir name
# each time you do a new experiment-- only if you change the
# configuration in a way that affects the tree.
tree_dir=exp/$dataset/chain${nnet3_affix}/tree_a_sp
# the 'lang' directory is created by this script.
# If you create such a directory with a non-standard topology
# you should probably name it differently.
lang=data/lang_chain

for f in $train_data_dir/feats.scp $train_ivector_dir/ivector_online.scp \
    $lores_train_data_dir/feats.scp $gmm_dir/final.mdl \
    $ali_dir/ali.1.gz $gmm_dir/final.mdl; do
  [ ! -f $f ] && echo "$0: expected file $f to exist" && exit 1
done


if [ $stage -le 14 ]; then
  echo "$0: creating lang directory $lang with chain-type topology"
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  if [ -d $lang ]; then
    if [ $lang/L.fst -nt data/lang/L.fst ]; then
      echo "$0: $lang already exists, not overwriting it; continuing"
    else
      echo "$0: $lang already exists and seems to be older than data/lang..."
      echo " ... not sure what to do.  Exiting."
      exit 1;
    fi
  else
    cp -r data/lang $lang
    silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
    nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
    # Use our special topology... note that later on may have to tune this
    # topology.
    steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
  fi
fi

if [ $stage -le 15 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" ${lores_train_data_dir} \
    data/lang $gmm_dir $lat_dir
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 16 ]; then
  # Build a tree using our new topology.  We know we have alignments for the
  # speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
  # those.  The num-leaves is always somewhat less than the num-leaves from
  # the GMM baseline.
  if [ -f $tree_dir/final.mdl ]; then
     echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
     exit 1;
  fi
  steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$train_cmd" 3500 ${lores_train_data_dir} \
    $lang $ali_dir $tree_dir
fi