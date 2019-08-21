[![Build Status](https://travis-ci.org/makcedward/nlpaug.svg?branch=master)](https://travis-ci.org/makcedward/nlpaug)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2d6d1d08016a4f78818161a89a2dfbfb)](https://www.codacy.com/app/makcedward/nlpaug?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=makcedward/nlpaug&amp;utm_campaign=Badge_Grade)
[![Codecov Badge](https://codecov.io/gh/makcedward/nlpaug/branch/master/graph/badge.svg)](https://codecov.io/gh/makcedward/nlpaug)

# nlpaug

This python library helps you with augmenting nlp for your machine learning projects. Visit this introduction to understand about [Data Augmentation in NLP](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28). `Augmenter` is the basic element of augmentation while `Flow` is a pipeline to orchestra multi augmenter together.
*   [Data Augmentation library for Text](https://towardsdatascience.com/data-augmentation-library-for-text-9661736b13ff)
*   [Data Augmentation library for Speech Recognition](https://towardsdatascience.com/data-augmentation-for-speech-recognition-e7c607482e78)
*   [Data Augmentation library for Audio](https://towardsdatascience.com/data-augmentation-for-audio-76912b01fdf6)
*   [Does your NLP model able to prevent adversarial attack?](https://medium.com/hackernoon/does-your-nlp-model-able-to-prevent-adversarial-attack-45b5ab75129c)

## Starter Guides
*   [Augmentation for character and word](https://github.com/makcedward/nlpaug/blob/master/example/overview.ipynb)
*   [Augmentation for spectrogram (audio input)](https://github.com/makcedward/nlpaug/blob/master/example/spectrogram_augmenter.ipynb)
*   [Augmentation for audio](https://github.com/makcedward/nlpaug/blob/master/example/audio_augmenter.ipynb)
*   How to train [TF-IDF model](https://github.com/makcedward/nlpaug/blob/master/example/tfidf-train_model.ipynb)
*   How to create [custom augmentation](https://github.com/makcedward/nlpaug/blob/master/example/custom_augmenter.ipynb)
*   [API Documentation](https://nlpaug.readthedocs.io/en/latest/)

## Augmenter
| Target | Augmenter | Action | Description |
|:---:|:---:|:---:|:---:|
| Character | RandomAug | insert | Insert character randomly |
| | | substitute | Substitute character randomly |
| | | swap | Swap character randomly |
| | | delete | Delete character randomly |
| | OcrAug | substitute | Simulate OCR engine error |
| | KeyboardAug | substitute | Simulate keyboard distance error |
| Word | RandomWordAug | swap | Swap word randomly |
| | | delete | Delete word randomly |
| | SpellingAug | substitute | Substitute word according to spelling mistake dictionary |
| | WordNetAug | substitute | Substitute word according to WordNet's synonym |
| | WordEmbsAug | insert | Insert word randomly from [word2vec](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a), [GloVe](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a) or [fasttext](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a) dictionary |
| | | substitute | Substitute word based on [word2vec](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a), [GloVe](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a) or [fasttext](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a) embeddings |
| | TfIdfAug | insert | Insert word randomly trained TF-IDF model |
| | | substitute | Substitute word based on TF-IDF score |
| | BertAug | insert | Insert word based by feeding surroundings word to [BERT](https://towardsdatascience.com/how-bert-leverage-attention-mechanism-and-transformer-to-learn-word-contextual-relations-5bbee1b6dbdb) language model |
| | | substitute | Substitute word based by feeding surroundings word to [BERT](https://towardsdatascience.com/how-bert-leverage-attention-mechanism-and-transformer-to-learn-word-contextual-relations-5bbee1b6dbdb) language model |
| Spectrogram | FrequencyMaskingAug | substitute | Set block of values to zero according to frequency dimension |
| | TimeMaskingAug | substitute | Set block of values to zero according to time dimension |
| Audio | NoiseAug | substitute | Inject noise |
| | PitchAug | substitute | Adjust audio's pitch |
| | ShiftAug | substitute | Shift time dimension forward/ backward |
| | SpeedAug | substitute | Adjust audio's speed |
| | CropAug | delete | Delete audio's segment |
| | LoudnessAug|substitute | Adjust audio's volume |
| | MaskAug | substitute | Mask audio's segment |

## Flow
| Pipeline | Description |
|:---:|:---:|
| Sequential | Apply list of augmentation functions sequentially |
| Sometimes | Apply some augmentation functions randomly |

## Installation

The library supports python 3.5+ in linux and window platform.

To install the library:
```bash
pip install nlpaug
```
or install the latest version (include BETA features) from github directly
```bash
pip install git+https://github.com/makcedward/nlpaug.git
```

If you use BertAug, install the following dependencies as well
```bash
pip install pytorch_pretrained_bert torch
```

If you use WordEmbsAug (word2vec, glove or fasttext), downloading pre-trained model first
```bash
from nlpaug.util.file.download import DownloadUtil
DownloadUtil.download_word2vec(dest_dir='.') # Download word2vec model
DownloadUtil.download_glove(model_name='glove.6B', dest_dir='.') # Download GloVe model
DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='.') # Download fasttext model
```

## Recent Changes

**BETA** Aug 16, 2019
*   Add new augmenter (CropAug, LoudnessAug, MaskAug)
*   QwertyAug is deprecated. It will be replaced by KeyboardAug
*   Remove StopWordsAug. It will be replaced by RandomWordAug
*   Code refactoring
*   Added model download function for word2vec, GloVe and fasttext

**0.0.6** Jul 29, 2019:
*   Added new augmenter [TF-IDF based word replacement augmenter](https://arxiv.org/pdf/1904.12848.pdf)(TfIdfAug)
*   Added new augmenter [Spelling mistake simulation augmenter](https://arxiv.org/pdf/1711.02173.pdf)(SpellingAug)
*   Added new augmenter [Stopword Dropout augmenter](https://arxiv.org/pdf/1809.02079.pdf)(StopWordsAug)
*   Fixed [#14](https://github.com/makcedward/nlpaug/issues/14)

See [changelog](https://github.com/makcedward/nlpaug/blob/master/CHANGE.md) for more details.

## Test

```
Word2vec, GloVe, Fasttext models are used in word insertion and substitution. Those model files are necessary in order to run test case. You have to add ".env" file in root directory and the content should be
    -   MODEL_DIR={MODEL FILE PATH}
```

```
Folder structure of model should be
    -- root directory
        - glove.6B.50d.txt
        - GoogleNews-vectors-negative300.bin
        - wiki-news-300d-1M.vec
```

## Research Reference
Some of the above augmenters are inspired by the following research papers. However, it does not always follow original implementation due to different reasons. If original implementation is needed, please refer to original source code.

| Augmenter | Inspired by|
|:---:|:---|
| RandomAug, SpellingAug | Y. Belinkov and Y. Bisk. [Synthetic and Natural Noise Both Break Neural Machine Translation](https://arxiv.org/pdf/1711.02173.pdf). 2017 |
| RandomAug | J. Ebrahimi, A. Rao, D. Lowd and D. Dou. [HotFlip: White-Box Adversarial Examples for Text Classification](https://arxiv.org/pdf/1712.06751.pdf). 2018 |
| RandomAug, RandomWordAug | J. Ebrahimi, D. Lowd and Dou. [On Adversarial Examples for Character-Level Neural Machine Translation](https://arxiv.org/pdf/1806.09030.pdf). 2018 |
| RandomAug, KeyboardAug | D. Pruthi, B. Dhingra and Z. C. Lipton. [Combating Adversarial Misspellings with Robust Word Recognition](https://arxiv.org/pdf/1905.11268.pdf). 2019 |
| RandomAug, RandomWordAug | T. Niu and M. Bansal. [Adversarial Over-Sensitivity and Over-Stability Strategies for Dialogue Models](https://arxiv.org/pdf/1809.02079.pdf). 2018 |
| RandomWordAug, WordNetAug | P. Minervini and S. Riedel. [Adversarially Regularising Neural NLI Models to Integrate Logical Background Knowledge](https://arxiv.org/pdf/1808.08609.pdf). 2018 |
| WordNetAug | X. Zhang, J. Zhao and Y. LeCun. [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf). 2015 |
| WordNetAug | S. Kobayashi and C. Coulombe. [Text Data Augmentation Made Simple By Leveraging NLP Cloud APIs](https://arxiv.org/ftp/arxiv/papers/1812/1812.04718.pdf). 2018 |
| TfIdfAug | Q. Xie, Z. Dai, E Hovy, M. T. Luong and Q. V. Le. [Unsupervised Data Augmentation](https://arxiv.org/pdf/1904.12848.pdf). 2019 |
| WordEmbsAug | W. Y. Wang and D. Yang. [That’s So Annoying!!!: A Lexical and Frame-Semantic Embedding Based Data Augmentation Approach to Automatic Categorization of Annoying Behaviors using #petpeeve Tweets](https://aclweb.org/anthology/D15-1306). 2015 |
| BertAug | S. Kobayashi. [Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relation](https://arxiv.org/pdf/1805.06201.pdf). 2018 |
| FrequencyMaskingAug, TimeMaskingAug | D. S. Park, W. Chan, Y. Zhang, C. C. Chiu, B. Zoph, E. D. Cubuk and Q. V. Le. [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/pdf/1904.08779.pdf). 2019 |

## Data Source
Capatured data from internet for building augmenter/ test case.

See [data source](https://github.com/makcedward/nlpaug/blob/master/SOURCE.md) for more details.