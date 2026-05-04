| Model | Features | Benchmark | Split | PR-AUC | AUROC | Brier | P@10% |
|-------|----------|-----------|-------|--------|-------|-------|-------|
| logistic | structured | strict | val | 0.5624 | 0.6422 | 0.2434 | 0.5882 |
| logistic | structured | strict | test | 0.5483 | 0.6077 | 0.2435 | 0.6667 |
| logistic | structured | intermediate | val | 0.3654 | 0.6728 | 0.2291 | 0.3529 |
| logistic | structured | intermediate | test | 0.3705 | 0.6757 | 0.2171 | 0.4167 |
| lightgbm | structured | strict | val | 0.5715 | 0.5787 | 0.3503 | 0.7647 |
| lightgbm | structured | strict | test | 0.5134 | 0.6096 | 0.3280 | 0.5333 |
| lightgbm | structured | intermediate | val | 0.3659 | 0.6400 | 0.2099 | 0.4118 |
| lightgbm | structured | intermediate | test | 0.3588 | 0.6908 | 0.1751 | 0.4167 |
| logistic | text | strict | val | 0.6969 | 0.7241 | 0.2141 | 0.8235 |
| logistic | text | strict | test | 0.6419 | 0.6822 | 0.2214 | 0.8667 |
| logistic | fusion | strict | val | 0.6338 | 0.6870 | 0.2275 | 0.7647 |
| logistic | fusion | strict | test | 0.5904 | 0.6581 | 0.2319 | 0.6667 |