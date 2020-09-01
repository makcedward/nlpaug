<p align="center">
    <br>
    <img src="https://github.com/makcedward/nlpaug/blob/master/res/logo_small.png"/>
    <br>
<p>
<p align="center">
    <a href="https://travis-ci.org/makcedward/nlpaug">
        <img alt="Build" src="https://travis-ci.org/makcedward/nlpaug.svg?branch=master">
    </a>
    <a href="https://www.codacy.com/app/makcedward/nlpaug?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=makcedward/nlpaug&amp;utm_campaign=Badge_Grade">
        <img alt="Code Quality" src="https://api.codacy.com/project/badge/Grade/2d6d1d08016a4f78818161a89a2dfbfb">
    </a>
</p>

# nlpaug

This python library helps you with augmenting nlp for your machine learning projects. Visit this introduction to understand about [Data Augmentation in NLP](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28). `Augmenter` is the basic element of augmentation while `Flow` is a pipeline to orchestra multi augmenter together.

## Features
*   Generate synthetic data for improving model performance without manual effort
*   Simple, easy-to-use and lightweight library. Augment data in 3 lines of code
*   Plug and play to any neural network frameworks (e.g. PyTorch, TensorFlow)
*   Support textual and audio input

## Book cited nlpaug
*   S. Vajjala, B. Majumder, A. Gupta and H. Surana. [Practical Natural Language Processing: A Comprehensive Guide to Building Real-World NLP Systems](https://www.amazon.com/Practical-Natural-Language-Processing-Pragmatic/dp/1492054054). 2020

## Research paper cited nlpaug
*   M. Raghu and  E. Schmidt. [A Survey of Deep Learning for Scientific Discovery](https://arxiv.org/pdf/2003.11755.pdf). 2020
*   H. Guan, J. Li, H. Xu and M. Devarakonda. [Robustly Pre-trained Neural Model for Direct Temporal Relation Extraction](https://arxiv.org/ftp/arxiv/papers/2004/2004.06216.pdf). 2020
*   X. He, K. Zhao and X. [Chu. AutoML: A Survey of the State-of-the-Art](https://arxiv.org/pdf/1908.00709.pdf). 2020
*   S. Illium, R. Muller, A. Sedlmeier and C. Linnhoff-Popien. [Surgical Mask Detection with Convolutional Neural Networks and Data Augmentations on Spectrograms](https://arxiv.org/pdf/2008.04590.pdf). 2020
*   D. Niederhut. [A Python package for text data enrichment](https://www.theoj.org/joss-papers/joss.02136/10.21105.joss.02136.pdf). 2020
*   P. Ryan, S. Takafuji, C. Yang, N. Wilson and C. McBride. [Using Self-Supervised Learning of Birdsong for Downstream Industrial Audio Classification](https://openreview.net/pdf?id=_P9LyJ5pMDb). 2020
*   Z. Shao, J. Yang and S. Ren. [Calibrating Deep Neural Network Classifiers on Out-of-Distribution Datasets](https://arxiv.org/pdf/2006.08914.pdf). 2020
*   S. Qiu, B. Xu, J. Zhang, Y. Wang, X. Shen, G. D. Melo, C. Long and X. Li EasyAug: An Automatic Textual Data Augmentation Platform for Classification Tasks. 2020
*   D. Nguyen, Q. H. Nguyen, M. Dao, D. Dang-Nguyen, C. Gurrin and B. T. Nguyen. [Duplicate Identification Algorithms in SaaS Platforms](http://doras.dcu.ie/24667/1/3379174.3392319.pdf). 2020

## Project cited nlpaug
*   D. Garcia-Olano and A. Jain. [Generating Counterfactual Explanations using Reinforcement Learning Methods for Tabular and Text data](http://www.diegoolano.com/files/RL_course_Fall_2019_Final_Project.pdf). 2019
*   L. Yi. [Avengers: Achieving Superhuman Performance for Question Answering on SQuAD 2.0 Using Multiple Data Augmentations, Randomized Mini-Batch Training and Architecture Ensembling](https://pdfs.semanticscholar.org/ce36/6e8f69a26ea84a65fc2b37d7492f6c8993fe.pdf). 2020

<h3 align="center">Textual Data Augmentation Example</h3>
<br><p align="center"><img src="https://github.com/makcedward/nlpaug/blob/master/res/textual_example.png"/></p>
<h3 align="center">Acoustic Data Augmentation Example</h3>
<br><p align="center"><img src="https://github.com/makcedward/nlpaug/blob/master/res/audio_example.png"/></p>
    
| Section | Description |
|:---:|:---:|
| [Quick Demo](https://github.com/makcedward/nlpaug#quick-demo) | How to use this library |
| [Augmenter](https://github.com/makcedward/nlpaug#augmenter) | Introduce all available augmentation methods |
| [Installation](https://github.com/makcedward/nlpaug#installation) | How to install this library |
| [Recent Changes](https://github.com/makcedward/nlpaug#recent-changes) | Latest enhancement |
| [Extension Reading](https://github.com/makcedward/nlpaug#extension-reading) | More real life examples or researchs |
| [Reference](https://github.com/makcedward/nlpaug#reference) | Refernce of external resources such as data or model |

## Quick Demo
*   [Example of Augmentation for Textual Inputs](https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb)
*   [Example of Augmentation for Multilingual Textual Inputs ](https://github.com/makcedward/nlpaug/blob/master/example/textual_language_augmenter.ipynb)
*   [Example of Augmentation for Spectrogram Inputs](https://github.com/makcedward/nlpaug/blob/master/example/spectrogram_augmenter.ipynb)
*   [Example of Augmentation for Audio Inputs](https://github.com/makcedward/nlpaug/blob/master/example/audio_augmenter.ipynb)
*   [Example of Orchestra Multiple Augmenters](https://github.com/makcedward/nlpaug/blob/master/example/flow.ipynb)
*   [Example of Showing Augmentation History](https://github.com/makcedward/nlpaug/blob/master/example/change_log.ipynb)
*   How to train [TF-IDF model](https://github.com/makcedward/nlpaug/blob/master/example/tfidf-train_model.ipynb)
*   How to create [custom augmentation](https://github.com/makcedward/nlpaug/blob/master/example/custom_augmenter.ipynb)
*   [API Documentation](https://nlpaug.readthedocs.io/en/latest/)

## Augmenter
| Augmenter | Target | Augmenter | Action | Description |
|:---:|:---:|:---:|:---:|:---:|
|Textual| Character | KeyboardAug | substitute | Simulate keyboard distance error |
|Textual| | OcrAug | substitute | Simulate OCR engine error |
|Textual| | [RandomAug](https://medium.com/hackernoon/does-your-nlp-model-able-to-prevent-adversarial-attack-45b5ab75129c) | insert, substitute, swap, delete | Apply augmentation randomly |
|Textual| Word | AntonymAug | substitute | Substitute opposite meaning word according to WordNet antonym|
|Textual| | ContextualWordEmbsAug | insert, substitute | Feeding surroundings word to [BERT](https://towardsdatascience.com/how-bert-leverage-attention-mechanism-and-transformer-to-learn-word-contextual-relations-5bbee1b6dbdb), DistilBERT, [RoBERTa](https://medium.com/towards-artificial-intelligence/a-robustly-optimized-bert-pretraining-approach-f6b6e537e6a6) or [XLNet](https://medium.com/dataseries/why-does-xlnet-outperform-bert-da98a8503d5b) language model to find out the most suitlabe word for augmentation|
|Textual| | RandomWordAug | swap, crop, delete | Apply augmentation randomly |
|Textual| | SpellingAug | substitute | Substitute word according to spelling mistake dictionary |
|Textual| | SplitAug | split | Split one word to two words randomly|
|Textual| | SynonymAug | substitute | Substitute similar word according to WordNet/ PPDB synonym |
|Textual| | [TfIdfAug](https://medium.com/towards-artificial-intelligence/unsupervised-data-augmentation-6760456db143) | insert, substitute | Use TF-IDF to find out how word should be augmented |
|Textual| | WordEmbsAug | insert, substitute | Leverage  [word2vec](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a), [GloVe](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a) or [fasttext](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a) embeddings to apply augmentation|
|Textual| | [BackTranslationAug](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28) | substitute | Leverage two translation models for augmentation |
|Textual| Sentence | ContextualWordEmbsForSentenceAug | insert | Insert sentence according to [XLNet](https://medium.com/dataseries/why-does-xlnet-outperform-bert-da98a8503d5b), [GPT2](https://towardsdatascience.com/too-powerful-nlp-model-generative-pre-training-2-4cc6afb6655) or DistilGPT2 prediction |
|Textual| | AbstSummAug | substitute | Summarize article by abstractive summarization method |
|Signal| Audio | CropAug | delete | Delete audio's segment |
|Signal| | LoudnessAug|substitute | Adjust audio's volume |
|Signal| | MaskAug | substitute | Mask audio's segment |
|Signal| | NoiseAug | substitute | Inject noise |
|Signal| | PitchAug | substitute | Adjust audio's pitch |
|Signal| | ShiftAug | substitute | Shift time dimension forward/ backward |
|Signal| | SpeedAug | substitute | Adjust audio's speed |
|Signal| | VtlpAug | substitute | Change vocal tract |
|Signal| Spectrogram | FrequencyMaskingAug | substitute | Set block of values to zero according to frequency dimension |
|Signal| | TimeMaskingAug | substitute | Set block of values to zero according to time dimension |

## Flow
| Augmenter | Augmenter | Description |
|:---:|:---:|:---:|
|Pipeline| Sequential | Apply list of augmentation functions sequentially |
|Pipeline| Sometimes | Apply some augmentation functions randomly |

## Installation
The library supports python 3.5+ in linux and window platform.

To install the library:
```bash
pip install nlpaug numpy requests
```
or install the latest version (include BETA features) from github directly
```bash
pip install git+https://github.com/makcedward/nlpaug.git numpy
```

If you use ContextualWordEmbsAug, ContextualWordEmbsForSentenceAug and AbstSummAug, install the following dependencies as well
```bash
pip install torch>=1.6.0 transformers>=3.0.2
```

If you use BackTranslationAug, install the following dependencies as well
```bash
pip install torch>=1.6.0 fairseq>=0.9.0
```

If you use AntonymAug, SynonymAug, install the following dependencies as well
```bash
pip install nltk>=3.4.5
```

If you use WordEmbsAug (word2vec, glove or fasttext), downloading pre-trained model first
```bash
from nlpaug.util.file.download import DownloadUtil
DownloadUtil.download_word2vec(dest_dir='.') # Download word2vec model
DownloadUtil.download_glove(model_name='glove.6B', dest_dir='.') # Download GloVe model
DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='.') # Download fasttext model
```

If you use SynonymAug (PPDB), downloading file from the following URI. You may not able to run the augmenter if you get PPDB file from other website
```bash
http://paraphrase.org/#/download
```

If you use PitchAug, SpeedAug and VtlpAug, install the following dependencies as well
```bash
pip install librosa>=0.7.1 matplotlib
```

## Recent Changes

### 1.0.0dev Sep, 2020
*   Upgraded to use AutoModel and AutoTokeizer for ContextualWordEmbsAug, ContextualWordEmbsForSentenceAug and AbstSummAug. Fix [#133](https://github.com/makcedward/nlpaug/issues/133), [#105]((https://github.com/makcedward/nlpaug/issues/105))
*   Refactoring audio and spectrogram augmenters
*   Added LoudnessAug into spectrogram augmenters

See [changelog](https://github.com/makcedward/nlpaug/blob/master/CHANGE.md) for more details.

## Extension Reading
*   [Data Augmentation library for Text](https://towardsdatascience.com/data-augmentation-library-for-text-9661736b13ff)
*   [Does your NLP model able to prevent adversarial attack?](https://medium.com/hackernoon/does-your-nlp-model-able-to-prevent-adversarial-attack-45b5ab75129c)
*   [How does Data Noising Help to Improve your NLP Model?](https://medium.com/towards-artificial-intelligence/how-does-data-noising-help-to-improve-your-nlp-model-480619f9fb10)
*   [Data Augmentation library for Speech Recognition](https://towardsdatascience.com/data-augmentation-for-speech-recognition-e7c607482e78)
*   [Data Augmentation library for Audio](https://towardsdatascience.com/data-augmentation-for-audio-76912b01fdf6)
*   [Unsupervied Data Augmentation](https://medium.com/towards-artificial-intelligence/unsupervised-data-augmentation-6760456db143)
*   [A Visual Survey of Data Augmentation in NLP](https://amitness.com/2020/05/data-augmentation-for-nlp/)


## Reference
This library uses data (e.g. capturing from internet), research (e.g. following augmenter idea), model (e.g. using pre-trained model) See [data source](https://github.com/makcedward/nlpaug/blob/master/SOURCE.md) for more details.

## Citing

```latex
@misc{ma2019nlpaug,
  title={NLP Augmentation},
  author={Edward Ma},
  howpublished={https://github.com/makcedward/nlpaug},
  year={2019}
}
```

## Contributions (Supporting Other Languages)
- [sakares](https://github.com/sakares): Add Thai support to KeyboardAug
