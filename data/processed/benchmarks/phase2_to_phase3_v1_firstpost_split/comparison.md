| Model | Features | Benchmark | Split | PR-AUC | AUROC | Brier | P@10% | Threshold (src) |
|-------|----------|-----------|-------|--------|-------|-------|-------|-----------------|
| logistic | structured | strict | val | 0.6022 | 0.6478 | 0.2385 | 0.5882 | 0.448 (self) |
| logistic | structured | strict | test | 0.5935 [0.499, 0.691] | 0.6499 [0.558, 0.734] | 0.2329 | 0.7333 | 0.448 (validation) |
| logistic | structured | intermediate | val | 0.3620 | 0.6782 | 0.2252 | 0.3889 | 0.426 (self) |
| logistic | structured | intermediate | test | 0.3889 [0.289, 0.516] | 0.6872 [0.609, 0.765] | 0.2104 | 0.4857 | 0.426 (validation) |
| lightgbm | structured | strict | val | 0.6390 | 0.6101 | 0.3317 | 0.8824 | 0.010 (self) |
| lightgbm | structured | strict | test | 0.5986 [0.500, 0.705] | 0.6484 [0.560, 0.737] | 0.3049 | 0.8667 | 0.010 (validation) |
| lightgbm | structured | intermediate | val | 0.3377 | 0.6497 | 0.2012 | 0.4167 | 0.097 (self) |
| lightgbm | structured | intermediate | test | 0.3281 [0.241, 0.443] | 0.6625 [0.588, 0.738] | 0.1728 | 0.3714 | 0.097 (validation) |
| logistic | text | strict | val | 0.6498 | 0.6998 | 0.2219 | 0.6471 | 0.484 (self) |
| logistic | text | strict | test | 0.6273 [0.535, 0.731] | 0.6744 [0.588, 0.752] | 0.2279 | 0.8 | 0.484 (validation) |
| logistic | text | intermediate | val | 0.4678 | 0.7512 | 0.1929 | 0.5833 | 0.479 (self) |
| logistic | text | intermediate | test | 0.3864 [0.307, 0.510] | 0.7248 [0.658, 0.799] | 0.1957 | 0.5143 | 0.479 (validation) |
| logistic | text | broad_filtered | val | 0.2771 | 0.7303 | 0.1805 | 0.2857 | 0.542 (self) |
| logistic | text | broad_filtered | test | 0.2560 [0.197, 0.360] | 0.7251 [0.653, 0.792] | 0.1891 | 0.3509 | 0.542 (validation) |
| logistic | fusion | strict | val | 0.6344 | 0.6691 | 0.2331 | 0.6471 | 0.394 (self) |
| logistic | fusion | strict | test | 0.6055 [0.513, 0.710] | 0.6800 [0.591, 0.762] | 0.2284 | 0.6667 | 0.394 (validation) |
| logistic | fusion | intermediate | val | 0.4594 | 0.7304 | 0.1997 | 0.5556 | 0.541 (self) |
| logistic | fusion | intermediate | test | 0.3993 [0.312, 0.522] | 0.7564 [0.697, 0.819] | 0.1885 | 0.4571 | 0.541 (validation) |