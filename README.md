[![Build Status](https://travis-ci.org/makcedward/nlpaug.svg?branch=master)](https://travis-ci.org/makcedward/nlpaug)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2d6d1d08016a4f78818161a89a2dfbfb)](https://www.codacy.com/app/makcedward/nlpaug?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=makcedward/nlpaug&amp;utm_campaign=Badge_Grade)
[![Codecov Badge](https://codecov.io/gh/makcedward/nlpaug/branch/master/graph/badge.svg)](https://codecov.io/gh/makcedward/nlpaug)

# nlpaug

This python library helps you with augmenting nlp for your machine learning projects. Visit this introduction to understand about [Data Augmentation in NLP](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28)

## Feature

* Provide both character and word level augmentations which include:
    * Character Augmentation: OCR, QWERTY(Keyboard Distance), Random Behavior
    * Word Augmentation:
        * Random Behavior: RandomWord
        * Synonym: WordNet
        * Word Embeddings: [word2vec, GloVe, fasttext](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a),
        * Language Models: [BERT](https://towardsdatascience.com/how-bert-leverage-attention-mechanism-and-transformer-to-learn-word-contextual-relations-5bbee1b6dbdb)
    * Speech Recognition Augmentation:
        * Spectrogram: Frequency Masking, Time Masking
* Flow orchestration is supported. Flow includes:
    * Sequential: Apply data augmentations one by one
    * Sometimes: Apply some augmentations randomly

## Example
* How to use [pre-defined augmentation](https://github.com/makcedward/nlpaug/blob/master/example/overview.ipynb)
* How to create [custom augmentation](https://github.com/makcedward/nlpaug/blob/master/example/custom_augmenter.ipynb)
* How to use [spectrogram augmentation for speech recognition](https://github.com/makcedward/nlpaug/blob/master/example/spectrogram_augmenter.ipynb)

Frequency Masking
![Frequency Masking](https://github.com/makcedward/nlpaug/blob/master/res/spectrogram-frequency_masking.png)

Time Masking
![Frequency Masking](https://github.com/makcedward/nlpaug/blob/master/res/spectrogram-time_masking.png)

## Installation

The library supports python 3.5+ in linux and window platform.

To install the library:
```bash
pip install nlpaug
```

Download word2vec or GloVe files if you use `Word2VecAug` or `GloVeAug`:
* word2vec([GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/))
* GloVe([glove.6B.50d](https://nlp.stanford.edu/projects/glove/))
* fasttext([wiki-news-300d-1M.vec.zip](https://fasttext.cc/docs/en/english-vectors.html))

## Recent Changes

**0.0.1** Mar 20, 2019: Project initialization

**0.0.2** Apr 30, 2019: Added Frequency Masking and Time Masking for Speech Recognition (Spectrogram). Added librosa library dependency for converting wav to spectrogram.

## Test

```
Word2vec and GloVe models are used in word insertion and substitution. Those model files are necessary in order to run test case. You have to add ".env" file in root directory and the content should be
	- MODEL_DIR={MODEL FILE PATH}
```

```
Folder structure of model should be
	-- root directory
		- glove.6B.50d.txt
		- GoogleNews-vectors-negative300.bin
		- wiki-news-300d-1M.vec
```