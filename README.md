[![Build Status](https://travis-ci.org/makcedward/nlpaug.svg?branch=master)](https://travis-ci.org/makcedward/nlpaug)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2d6d1d08016a4f78818161a89a2dfbfb)](https://www.codacy.com/app/makcedward/nlpaug?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=makcedward/nlpaug&amp;utm_campaign=Badge_Grade)
[![Codecov Badge](https://codecov.io/gh/makcedward/nlpaug/branch/master/graph/badge.svg)](https://codecov.io/gh/makcedward/nlpaug)

# nlpaug

This python library helps you with augmenting nlp for your machine learning projects.

## Feature

* Provide both character and word level augmentations which include:
    * Character Augmentation: OCR, QWERTY(Keyboard Distance), Random Behavior
    * Word Augmentation: word2vec, GloVe, WordNet, Random Behavior
* Flow orchestration is supported. Flow includes:
    * Sequential: Apply data augmentations one by one
    * Sometimes: Apply some augmentations randomly

## Installation

The library supports python 3.5+ in linux and window platform.

To install the library:
```bash
pip install nlpaug
```

Download word2vec or GloVe files if you use `Word2VecAug` or `GloVeAug`:
* word2vec([GoogleNews-vectors-negative300](https://code.google.com/archive/p/word2vec/))
* GloVe([glove.6B.50d](https://nlp.stanford.edu/projects/glove/))

## Recent Changes

**0.0.1**: Project initialization (Mar 20, 2019)


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
```