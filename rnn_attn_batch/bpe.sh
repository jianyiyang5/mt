#!/usr/bin/env bash
cut -f1 data/eng-fra.raw > data/eng.txt
cut -f2 data/eng-fra.raw > data/fra.txt
subword-nmt learn-bpe -s 13000 < data/eng.txt > data/eng.dict
subword-nmt apply-bpe -c data/eng.dict < data/eng.txt > data/eng.bpe
subword-nmt learn-bpe -s 13000 < data/fra.txt > data/fra.dict
subword-nmt apply-bpe -c data/fra.dict < data/fra.txt > data/fra.bpe
paste data/eng.bpe data/fra.bpe > data/eng-fra.bpe
