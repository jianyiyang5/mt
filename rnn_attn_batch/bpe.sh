#!/usr/bin/env bash
cut -f1 data/eng-fra.raw > eng.txt
cut -f2 data/eng-fra.raw > fra.txt
subword-nmt learn-bpe -s 13000 < eng.txt > eng.dict
subword-nmt apply-bpe -c eng.dict < eng.txt > eng.bpe
subword-nmt learn-bpe -s 13000 < fra.txt > fra.dict
subword-nmt apply-bpe -c fra.dict < fra.txt > fra.bpe
paste eng.bpe fra.bpe > eng-fra.bpe
