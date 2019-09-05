#!/usr/bin/env bash
set -ex
pip install subword-nmt
bash bpe.sh
python split_data.py
python train.py
python eval.py
perl multi-bleu.perl test/test.ref < test/test.hyp