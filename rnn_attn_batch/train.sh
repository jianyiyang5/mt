#!/usr/bin/env bash
set -ex
python split_data.py
python train.py
python eval.py
perl multi-bleu.perl test/test.en < test/test.hyp