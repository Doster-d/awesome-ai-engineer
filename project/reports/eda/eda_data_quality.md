# Разведочный анализ данных

## 1. Назначение наборов данных

В проекте используются два набора данных.

CUAD используется для задачи извлечения юридически значимых условий из договоров. В CUAD отдельно рассматриваются:
- `*-Answer` поля — классификационные метки наличия условия;
- текстовые clause/evidence-поля — фрагменты договора для извлечения доказательств.

ContractNLI используется для задачи проверки утверждений по тексту договора с опорой на evidence-фрагменты.

## 2. Общая сводка

| metric                                                   |        value |
|:---------------------------------------------------------|-------------:|
| cuad_documents_txt                                       |   510        |
| cuad_master_rows                                         |   510        |
| cuad_master_columns                                      |    83        |
| cuad_answer_columns                                      |    41        |
| cuad_text_clause_columns                                 |    37        |
| cuad_text_clause_examples                                |  9204        |
| cuad_clause_match_rate                                   |     0.899718 |
| cuad_empty_text_files                                    |     0        |
| cuad_median_words                                        |  4971        |
| cuad_p95_words                                           | 24853.2      |
| contractnli_documents                                    |   607        |
| contractnli_examples                                     | 10319        |
| contractnli_train_docs                                   |   423        |
| contractnli_dev_docs                                     |    61        |
| contractnli_test_docs                                    |   123        |
| contractnli_entailment_examples                          |  5017        |
| contractnli_contradiction_examples                       |  1156        |
| contractnli_not_mentioned_examples                       |  4146        |
| contractnli_examples_without_evidence                    |  4146        |
| contractnli_entailment_or_contradiction_without_evidence |     0        |
| contractnli_median_words                                 |  1498        |
| contractnli_p95_words                                    |  3701.4      |

## 3. CUAD

Количество TXT-договоров: 510.

Количество строк в `master_clauses.csv`: 510.

Количество `*-Answer` полей: 41.

Количество текстовых clause/evidence-полей: 37.

### Длины договоров CUAD

| metric        |   count |      mean |       std |   min |      25% |     50% |     75% |    max |
|:--------------|--------:|----------:|----------:|------:|---------:|--------:|--------:|-------:|
| chars         |     510 | 50632.2   | 53947.3   |   645 | 16149.5  | 32141.5 | 65339.2 | 300304 |
| words         |     510 |  7785.4   |  8235.3   |   109 |  2436.75 |  4971   | 10066.8 |  47642 |
| approx_tokens |     510 | 12658.4   | 13486.8   |   162 |  4038    |  8036   | 16335   |  75076 |
| lines         |     510 |   290.918 |   409.132 |     1 |    59    |   144   |   353.5 |   4027 |

### Текстовые clause/evidence-поля CUAD

| clause_column                      |   documents_with_text |   span_count |   documents_with_text_share |   median_text_words |
|:-----------------------------------|----------------------:|-------------:|----------------------------:|--------------------:|
| Document Name                      |                   510 |          521 |                   1         |                 3   |
| Governing Law                      |                   437 |          457 |                   0.856863  |                29   |
| Anti-Assignment                    |                   374 |          612 |                   0.733333  |                37   |
| Cap On Liability                   |                   275 |          632 |                   0.539216  |                52   |
| License Grant                      |                   255 |          716 |                   0.5       |                57   |
| Audit Rights                       |                   214 |          613 |                   0.419608  |                41   |
| Termination For Convenience        |                   183 |          223 |                   0.358824  |                31   |
| Post-Termination Services          |                   182 |          409 |                   0.356863  |                60   |
| Exclusivity                        |                   180 |          388 |                   0.352941  |                47.5 |
| Renewal Term                       |                   176 |          199 |                   0.345098  |                46   |
| Insurance                          |                   166 |          531 |                   0.32549   |                39   |
| Revenue/Profit Sharing             |                   166 |          395 |                   0.32549   |                45   |
| Minimum Commitment                 |                   165 |          392 |                   0.323529  |                45   |
| Non-Transferable License           |                   138 |          282 |                   0.270588  |                52.5 |
| Ip Ownership Assignment            |                   124 |          296 |                   0.243137  |                55   |
| Change Of Control                  |                   121 |          213 |                   0.237255  |                56   |
| Non-Compete                        |                   119 |          226 |                   0.233333  |                53.5 |
| Uncapped Liability                 |                   111 |          151 |                   0.217647  |                71   |
| Notice Period To Terminate Renewal |                   111 |          117 |                   0.217647  |                49   |
| Covenant Not To Sue                |                   100 |          159 |                   0.196078  |                55   |
| Rofr/Rofo/Rofn                     |                    85 |          350 |                   0.166667  |                54   |
| Volume Restriction                 |                    82 |          165 |                   0.160784  |                37   |
| Competitive Restriction Exception  |                    76 |          114 |                   0.14902   |                59   |
| Warranty Duration                  |                    75 |          164 |                   0.147059  |                43   |
| Irrevocable Or Perpetual License   |                    70 |          150 |                   0.137255  |                73   |
| Liquidated Damages                 |                    61 |          113 |                   0.119608  |                52   |
| Affiliate License-Licensee         |                    59 |          104 |                   0.115686  |                79.5 |
| No-Solicit Of Employees            |                    59 |           75 |                   0.115686  |                66   |
| Joint Ip Ownership                 |                    46 |          100 |                   0.0901961 |                49   |
| Non-Disparagement                  |                    38 |           54 |                   0.0745098 |                46   |

### Answer-поля CUAD

| base_clause                       |   yes_count |   no_count |   yes_share |   no_share |
|:----------------------------------|------------:|-----------:|------------:|-----------:|
| Anti-Assignment                   |         374 |        136 |   0.733333  |   0.266667 |
| Cap On Liability                  |         275 |        235 |   0.539216  |   0.460784 |
| License Grant                     |         255 |        255 |   0.5       |   0.5      |
| Audit Rights                      |         214 |        296 |   0.419608  |   0.580392 |
| Termination For Convenience       |         183 |        327 |   0.358824  |   0.641176 |
| Post-Termination Services         |         182 |        328 |   0.356863  |   0.643137 |
| Exclusivity                       |         180 |        330 |   0.352941  |   0.647059 |
| Insurance                         |         167 |        343 |   0.327451  |   0.672549 |
| Revenue/Profit Sharing            |         166 |        344 |   0.32549   |   0.67451  |
| Minimum Commitment                |         165 |        345 |   0.323529  |   0.676471 |
| Non-Transferable License          |         138 |        372 |   0.270588  |   0.729412 |
| Ip Ownership Assignment           |         124 |        386 |   0.243137  |   0.756863 |
| Change Of Control                 |         121 |        389 |   0.237255  |   0.762745 |
| Non-Compete                       |         119 |        391 |   0.233333  |   0.766667 |
| Uncapped Liability                |         111 |        399 |   0.217647  |   0.782353 |
| Covenant Not To Sue               |         100 |        410 |   0.196078  |   0.803922 |
| Rofr/Rofo/Rofn                    |          85 |        425 |   0.166667  |   0.833333 |
| Volume Restriction                |          82 |        428 |   0.160784  |   0.839216 |
| Competitive Restriction Exception |          76 |        434 |   0.14902   |   0.85098  |
| Warranty Duration                 |          75 |        435 |   0.147059  |   0.852941 |
| Irrevocable Or Perpetual License  |          70 |        440 |   0.137255  |   0.862745 |
| Liquidated Damages                |          61 |        449 |   0.119608  |   0.880392 |
| Affiliate License-Licensee        |          59 |        451 |   0.115686  |   0.884314 |
| No-Solicit Of Employees           |          59 |        451 |   0.115686  |   0.884314 |
| Joint Ip Ownership                |          46 |        464 |   0.0901961 |   0.909804 |
| Non-Disparagement                 |          38 |        472 |   0.0745098 |   0.92549  |
| No-Solicit Of Customers           |          34 |        476 |   0.0666667 |   0.933333 |
| Third Party Beneficiary           |          33 |        477 |   0.0647059 |   0.935294 |
| Most Favored Nation               |          28 |        482 |   0.054902  |   0.945098 |
| Affiliate License-Licensor        |          23 |        487 |   0.045098  |   0.954902 |

### Согласованность Answer и текстовых полей

| clause                             |   documents_with_text |   answer_yes_count |   text_when_answer_yes |   answer_yes_without_text |   text_without_answer_yes |
|:-----------------------------------|----------------------:|-------------------:|-----------------------:|--------------------------:|--------------------------:|
| Document Name                      |                   510 |                  0 |                      0 |                         0 |                       510 |
| Governing Law                      |                   437 |                  0 |                      0 |                         0 |                       437 |
| Anti-Assignment                    |                   374 |                374 |                    374 |                         0 |                         0 |
| Cap On Liability                   |                   275 |                275 |                    275 |                         0 |                         0 |
| License Grant                      |                   255 |                255 |                    255 |                         0 |                         0 |
| Audit Rights                       |                   214 |                214 |                    214 |                         0 |                         0 |
| Termination For Convenience        |                   183 |                183 |                    183 |                         0 |                         0 |
| Post-Termination Services          |                   182 |                182 |                    182 |                         0 |                         0 |
| Exclusivity                        |                   180 |                180 |                    180 |                         0 |                         0 |
| Renewal Term                       |                   176 |                  0 |                      0 |                         0 |                       176 |
| Insurance                          |                   166 |                167 |                    166 |                         1 |                         0 |
| Revenue/Profit Sharing             |                   166 |                166 |                    166 |                         0 |                         0 |
| Minimum Commitment                 |                   165 |                165 |                    165 |                         0 |                         0 |
| Non-Transferable License           |                   138 |                138 |                    138 |                         0 |                         0 |
| Ip Ownership Assignment            |                   124 |                124 |                    124 |                         0 |                         0 |
| Change Of Control                  |                   121 |                121 |                    121 |                         0 |                         0 |
| Non-Compete                        |                   119 |                119 |                    119 |                         0 |                         0 |
| Notice Period To Terminate Renewal |                   111 |                  0 |                      0 |                         0 |                       111 |
| Uncapped Liability                 |                   111 |                111 |                    111 |                         0 |                         0 |
| Covenant Not To Sue                |                   100 |                100 |                    100 |                         0 |                         0 |
| Rofr/Rofo/Rofn                     |                    85 |                 85 |                     85 |                         0 |                         0 |
| Volume Restriction                 |                    82 |                 82 |                     82 |                         0 |                         0 |
| Competitive Restriction Exception  |                    76 |                 76 |                     76 |                         0 |                         0 |
| Warranty Duration                  |                    75 |                 75 |                     75 |                         0 |                         0 |
| Irrevocable Or Perpetual License   |                    70 |                 70 |                     70 |                         0 |                         0 |
| Liquidated Damages                 |                    61 |                 61 |                     61 |                         0 |                         0 |
| Affiliate License-Licensee         |                    59 |                 59 |                     59 |                         0 |                         0 |
| No-Solicit Of Employees            |                    59 |                 59 |                     59 |                         0 |                         0 |
| Joint Ip Ownership                 |                    46 |                 46 |                     46 |                         0 |                         0 |
| Non-Disparagement                  |                    38 |                 38 |                     38 |                         0 |                         0 |

### Поиск текстовых clause-фрагментов в полном тексте договора

Общий match rate: 0.8997

| clause_type                        |   examples |   matched |   match_rate |   contract_found_rate |   median_clause_words |
|:-----------------------------------|-----------:|----------:|-------------:|----------------------:|----------------------:|
| License Grant                      |        716 |       640 |     0.893855 |              0.97067  |                  57   |
| Cap On Liability                   |        632 |       574 |     0.908228 |              0.966772 |                  52   |
| Audit Rights                       |        613 |       562 |     0.916803 |              0.959217 |                  41   |
| Anti-Assignment                    |        612 |       557 |     0.910131 |              0.970588 |                  37   |
| Insurance                          |        531 |       485 |     0.913371 |              0.967985 |                  39   |
| Document Name                      |        521 |       507 |     0.973129 |              0.973129 |                   3   |
| Governing Law                      |        457 |       439 |     0.960613 |              0.97593  |                  29   |
| Post-Termination Services          |        409 |       350 |     0.855746 |              0.94132  |                  60   |
| Revenue/Profit Sharing             |        395 |       361 |     0.913924 |              0.972152 |                  45   |
| Minimum Commitment                 |        392 |       355 |     0.905612 |              0.984694 |                  45   |
| Exclusivity                        |        388 |       353 |     0.909794 |              0.966495 |                  47.5 |
| Rofr/Rofo/Rofn                     |        350 |       306 |     0.874286 |              0.917143 |                  54   |
| Ip Ownership Assignment            |        296 |       263 |     0.888514 |              0.952703 |                  55   |
| Non-Transferable License           |        282 |       255 |     0.904255 |              0.950355 |                  52.5 |
| Non-Compete                        |        226 |       186 |     0.823009 |              0.955752 |                  53.5 |
| Termination For Convenience        |        223 |       194 |     0.869955 |              0.96861  |                  31   |
| Change Of Control                  |        213 |       171 |     0.802817 |              0.948357 |                  56   |
| Renewal Term                       |        199 |       185 |     0.929648 |              0.979899 |                  46   |
| Volume Restriction                 |        165 |       159 |     0.963636 |              1        |                  37   |
| Warranty Duration                  |        164 |       151 |     0.920732 |              0.981707 |                  43   |
| Covenant Not To Sue                |        159 |       141 |     0.886792 |              0.974843 |                  55   |
| Uncapped Liability                 |        151 |       131 |     0.86755  |              0.966887 |                  71   |
| Irrevocable Or Perpetual License   |        150 |       131 |     0.873333 |              0.973333 |                  73   |
| Notice Period To Terminate Renewal |        117 |       110 |     0.940171 |              0.974359 |                  49   |
| Competitive Restriction Exception  |        114 |       102 |     0.894737 |              0.964912 |                  59   |
| Liquidated Damages                 |        113 |       100 |     0.884956 |              0.955752 |                  52   |
| Affiliate License-Licensee         |        104 |        91 |     0.875    |              0.980769 |                  79.5 |
| Joint Ip Ownership                 |        100 |        89 |     0.89     |              0.99     |                  49   |
| No-Solicit Of Employees            |         75 |        56 |     0.746667 |              0.946667 |                  66   |
| Source Code Escrow                 |         63 |        60 |     0.952381 |              1        |                  40   |

## 4. ContractNLI

Количество документов: 607.

Количество примеров: 10319.

Разбиение по документам:

| split   |   documents |
|:--------|------------:|
| dev     |          61 |
| test    |         123 |
| train   |         423 |

Разбиение по примерам:

| split   |   examples |
|:--------|-----------:|
| dev     |       1037 |
| test    |       2091 |
| train   |       7191 |

### Распределение классов

| class         |   examples |
|:--------------|-----------:|
| entailment    |       5017 |
| not_mentioned |       4146 |
| contradiction |       1156 |

### Evidence по классам

| gold_label    |   examples |   mean_evidence_count |   median_evidence_count |   examples_without_evidence |   share_without_evidence |
|:--------------|-----------:|----------------------:|------------------------:|----------------------------:|-------------------------:|
| entailment    |       5017 |               1.94957 |                       2 |                           0 |                        0 |
| not_mentioned |       4146 |               0       |                       0 |                        4146 |                        1 |
| contradiction |       1156 |               1.89619 |                       1 |                           0 |                        0 |

### Матрица гипотез

| label_id   | label_short_description                        |   entailment |   contradiction |   not_mentioned |   total |
|:-----------|:-----------------------------------------------|-------------:|----------------:|----------------:|--------:|
| nda-1      | Explicit identification                        |          134 |             158 |             315 |     607 |
| nda-10     | Confidentiality of Agreement                   |          234 |               2 |             371 |     607 |
| nda-11     | No reverse engineering                         |           80 |               1 |             526 |     607 |
| nda-12     | Permissible development of similar information |          376 |               0 |             231 |     607 |
| nda-13     | Permissible acquirement of similar information |          454 |               0 |             153 |     607 |
| nda-15     | No licensing                                   |          452 |               0 |             155 |     607 |
| nda-16     | Return of confidential information             |          240 |               4 |             363 |     607 |
| nda-17     | Permissible copy                               |          124 |             109 |             374 |     607 |
| nda-18     | No solicitation                                |          132 |               0 |             475 |     607 |
| nda-19     | Survival of obligations                        |          422 |              15 |             170 |     607 |
| nda-2      | None-inclusion of non-technical information    |           35 |             441 |             131 |     607 |
| nda-20     | Permissible post-agreement possession          |          164 |             236 |             207 |     607 |
| nda-3      | Inclusion of verbally conveyed information     |          388 |               7 |             212 |     607 |
| nda-4      | Limited use                                    |          517 |              17 |              73 |     607 |
| nda-5      | Sharing with employees                         |          492 |              16 |              99 |     607 |
| nda-7      | Sharing with third-parties                     |          375 |             150 |              82 |     607 |
| nda-8      | Notice on compelled disclosure                 |          398 |               0 |             209 |     607 |

## 5. Выводы для архитектуры

1. ContractNLI пригоден для режима проверки утверждений: `договор + гипотеза → entailment / contradiction / not_mentioned + evidence`.
2. CUAD пригоден для режима извлечения условий, но требует аккуратной обработки: `*-Answer` нельзя смешивать с текстовыми evidence-полями.
3. Для CUAD дополнительно проверяется, находятся ли текстовые фрагменты из `master_clauses.csv` в полном TXT-договоре.
4. Договоры достаточно длинные, поэтому нужен поиск по фрагментам и отдельный механизм анализа длинного контекста.
5. Для оценки качества нужны отдельные метрики: классификация наличия условия, извлечение evidence, overlap доказательных фрагментов.

## 6. Сохранённые артефакты

- `/data/source/personal/awesome-ai-engineer/project/data/processed/eda/eda_summary.csv`
- `/data/source/personal/awesome-ai-engineer/project/data/processed/eda/cuad_documents.csv`
- `/data/source/personal/awesome-ai-engineer/project/data/processed/eda/cuad_text_clause_stats.csv`
- `/data/source/personal/awesome-ai-engineer/project/data/processed/eda/cuad_answer_stats.csv`
- `/data/source/personal/awesome-ai-engineer/project/data/processed/eda/cuad_clause_pair_stats.csv`
- `/data/source/personal/awesome-ai-engineer/project/data/processed/eda/cuad_text_annotation_density_by_document.csv`
- `/data/source/personal/awesome-ai-engineer/project/data/processed/eda/cuad_clause_examples_match.csv`
- `/data/source/personal/awesome-ai-engineer/project/data/processed/eda/cuad_clause_match_stats.csv`
- `/data/source/personal/awesome-ai-engineer/project/data/processed/eda/contractnli_documents.csv`
- `/data/source/personal/awesome-ai-engineer/project/data/processed/eda/contractnli_examples.csv`
- `/data/source/personal/awesome-ai-engineer/project/data/processed/eda/contractnli_examples_with_evidence.jsonl`
- `/data/source/personal/awesome-ai-engineer/project/data/processed/eda/contractnli_evidence_spans.csv`
- `/data/source/personal/awesome-ai-engineer/project/reports/eda/figures`
