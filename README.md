### To do list

- [x] LSTM+Attn+GPU, 8/8/2019
- [x] BLEU score, 8/10
  - BLEU = 43.14, 72.4/48.9/36.4/28.1 (BP=0.990, ratio=0.990, hyp_len=100959, ref_len=102002)
- [x] batch decode, 8/11
  - (cpu run)BLEU = 51.45, 77.2/56.7/44.9/36.5 (BP=0.994, ratio=0.994, hyp_len=101269, ref_len=101856)
  - (gpu run)BLEU = 44.16, 73.5/50.2/37.7/29.3 (BP=0.983, ratio=0.983, hyp_len=100277, ref_len=102002)
- [x] transformer
- [ ] bpe
- [ ] beam search
- [ ] multi gpu
- [ ] plot



- [ ] data iter
- [ ] en-zh, zh-en
- [ ] multi node
- [ ] larger batch size, accumulate gradient, large lr, half precision
- [ ] beam search pruning
- [ ] bottleneck layer
- [ ] avg attn
- [ ] shortlist
- [ ] data selection?
- [ ] common crawl?
- [ ] cnn