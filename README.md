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
    <a href="https://pepy.tech/badge/nlpaug">
        <img alt="Downloads" src="https://pepy.tech/badge/nlpaug">
    </a>
</p>

# nlpaug

This python library helps you with augmenting nlp for your machine learning projects. Visit this introduction to understand about [Data Augmentation in NLP](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28). `Augmenter` is the basic element of augmentation while `Flow` is a pipeline to orchestra multi augmenter together.

## Features
*   Generate synthetic data for improving model performance without manual effort
*   Simple, easy-to-use and lightweight library. Augment data in 3 lines of code
*   Plug and play to any machine leanring/ neural network frameworks (e.g. scikit-learn, PyTorch, TensorFlow)
*   Support textual and audio input

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
| [Reference](https://github.com/makcedward/nlpaug#reference) | Reference of external resources such as data or model |

## Quick Demo
*   [Quick Example](https://github.com/makcedward/nlpaug/blob/master/example/quick_example.ipynb)
*   [Example of Augmentation for Textual Inputs](https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb)
*   [Example of Augmentation for Multilingual Textual Inputs ](https://github.com/makcedward/nlpaug/blob/master/example/textual_language_augmenter.ipynb)
*   [Example of Augmentation for Spectrogram Inputs](https://github.com/makcedward/nlpaug/blob/master/example/spectrogram_augmenter.ipynb)
*   [Example of Augmentation for Audio Inputs](https://github.com/makcedward/nlpaug/blob/master/example/audio_augmenter.ipynb)
*   [Example of Orchestra Multiple Augmenters](https://github.com/makcedward/nlpaug/blob/master/example/flow.ipynb)
*   [Example of Showing Augmentation History](https://github.com/makcedward/nlpaug/blob/master/example/change_log.ipynb)
*   How to train [TF-IDF model](https://github.com/makcedward/nlpaug/blob/master/example/tfidf-train_model.ipynb)
*   How to train [LAMBADA model](https://github.com/makcedward/nlpaug/blob/master/example/lambada-train_model.ipynb)
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
|Textual| | ReservedAug | substitute | Replace reserved words |
|Textual| Sentence | ContextualWordEmbsForSentenceAug | insert | Insert sentence according to [XLNet](https://medium.com/dataseries/why-does-xlnet-outperform-bert-da98a8503d5b), [GPT2](https://towardsdatascience.com/too-powerful-nlp-model-generative-pre-training-2-4cc6afb6655) or DistilGPT2 prediction |
|Textual| | AbstSummAug | substitute | Summarize article by abstractive summarization method |
|Textual| | LambadaAug | substitute | Using language model to generate text and then using classification model to retain high quality results |
|Signal| Audio | CropAug | delete | Delete audio's segment |
|Signal| | LoudnessAug|substitute | Adjust audio's volume |
|Signal| | MaskAug | substitute | Mask audio's segment |
|Signal| | NoiseAug | substitute | Inject noise |
|Signal| | PitchAug | substitute | Adjust audio's pitch |
|Signal| | ShiftAug | substitute | Shift time dimension forward/ backward |
|Signal| | SpeedAug | substitute | Adjust audio's speed |
|Signal| | VtlpAug | substitute | Change vocal tract |
|Signal| | NormalizeAug | substitute | Normalize audio |
|Signal| | PolarityInverseAug | substitute | Swap positive and negative for audio |
|Signal| Spectrogram | FrequencyMaskingAug | substitute | Set block of values to zero according to frequency dimension |
|Signal| | TimeMaskingAug | substitute | Set block of values to zero according to time dimension |
|Signal| | LoudnessAug | substitute | Adjust volume |

## Flow
| Augmenter | Augmenter | Description |
|:---:|:---:|:---:|
|Pipeline| Sequential | Apply list of augmentation functions sequentially |
|Pipeline| Sometimes | Apply some augmentation functions randomly |

## Installation
The library supports python 3.5+ in linux and window platform.

To install the library:
```bash
pip install numpy requests nlpaug
```
or install the latest version (include BETA features) from github directly
```bash
pip install numpy git+https://github.com/makcedward/nlpaug.git
```
or install over conda
```bash
conda install -c makcedward nlpaug
```

If you use BackTranslationAug, ContextualWordEmbsAug, ContextualWordEmbsForSentenceAug and AbstSummAug, installing the following dependencies as well
```bash
pip install torch>=1.6.0 transformers>=4.11.3 sentencepiece
```

If you use LambadaAug, installing the following dependencies as well
```bash
pip install simpletransformers>=0.61.10
```

If you use AntonymAug, SynonymAug, installing the following dependencies as well
```bash
pip install nltk>=3.4.5
```

If you use WordEmbsAug (word2vec, glove or fasttext), downloading pre-trained model first and installing the following dependencies as well
```bash
from nlpaug.util.file.download import DownloadUtil
DownloadUtil.download_word2vec(dest_dir='.') # Download word2vec model
DownloadUtil.download_glove(model_name='glove.6B', dest_dir='.') # Download GloVe model
DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='.') # Download fasttext model

pip install gensim>=4.1.2
```

If you use SynonymAug (PPDB), downloading file from the following URI. You may not able to run the augmenter if you get PPDB file from other website
```bash
http://paraphrase.org/#/download
```

If you use PitchAug, SpeedAug and VtlpAug, installing the following dependencies as well
```bash
pip install librosa>=0.9.1 matplotlib
```

## Recent Changes

### 1.1.11 Jul 6, 2022
*   [Return list of output](https://github.com/makcedward/nlpaug/issues/302)
*   [Fix download util](https://github.com/makcedward/nlpaug/issues/301)
*   [Fix lambda label misalignment](https://github.com/makcedward/nlpaug/issues/295)
*   [Add language pack reference link for SynonymAug](https://github.com/makcedward/nlpaug/issues/289)


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

## Citation

```latex
@misc{ma2019nlpaug,
  title={NLP Augmentation},
  author={Edward Ma},
  howpublished={https://github.com/makcedward/nlpaug},
  year={2019}
}
```

This package is cited by many books, workshop and academic research papers (70+). Here are some of examples and you may visit [here](https://github.com/makcedward/nlpaug/blob/master/CITED.md) to get the full list.

### Workshops cited nlpaug
*   S. Vajjala. [NLP without a readymade labeled dataset](https://rpubs.com/vbsowmya/tmls2021) at [Toronto Machine Learning Summit, 2021](https://www.torontomachinelearning.com/). 2021

### Book cited nlpaug
*   S. Vajjala, B. Majumder, A. Gupta and H. Surana. [Practical Natural Language Processing: A Comprehensive Guide to Building Real-World NLP Systems](https://www.amazon.com/Practical-Natural-Language-Processing-Pragmatic/dp/1492054054). 2020
*   A. Bartoli and A. Fusiello. [Computer Vision–ECCV 2020 Workshops](https://books.google.com/books?hl=en&lr=lang_en&id=0rYREAAAQBAJ&oi=fnd&pg=PR7&dq=nlpaug&ots=88bPp5rhnY&sig=C2ue8Xxbu09l59nAMOcVxWYvvWM#v=onepage&q=nlpaug&f=false). 2020
*   L. Werra, L. Tunstall, and T. Wolf [Natural Language Processing with Transformers](https://www.amazon.com/Natural-Language-Processing-Transformers-Applications/dp/1098103246/ref=sr_1_3?crid=2CWBPA8QG0TRU&keywords=Natural+Language+Processing+with+Transformers&qid=1645646312&sprefix=natural+language+processing+with+transformers%2Caps%2C111&sr=8-3). 2022

### Research paper cited nlpaug
*   Google: M. Raghu and  E. Schmidt. [A Survey of Deep Learning for Scientific Discovery](https://arxiv.org/pdf/2003.11755.pdf). 2020
*   Sirius XM: E. Jing, K. Schneck, D. Egan and S. A. Waterman. [Identifying Introductions in Podcast Episodes from Automatically Generated Transcripts](https://arxiv.org/pdf/2110.07096.pdf). 2021
*   Salesforce Research: B. Newman, P. K. Choubey and N. Rajani. [P-adapters: Robustly Extracting Factual Information from Language Modesl with Diverse Prompts](https://arxiv.org/pdf/2110.07280.pdf). 2021
*   Salesforce Research: L. Xue, M. Gao, Z. Chen, C. Xiong and R. Xu. [Robustness Evaluation of Transformer-based Form Field Extractors via Form Attacks](https://arxiv.org/pdf/2110.04413.pdf). 2021


## Contributions
<table>
  <tr>
    <td align="center"><a href="https://github.com/sakares"><img src="https://avatars.githubusercontent.com/u/1306031" width="100px;" alt=""/><br /><sub><b>sakares saengkaew</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/bdalal"><img src="https://avatars.githubusercontent.com/u/3478378?s=400&v=4" width="100px;" alt=""/><br /><sub><b>Binoy Dalal</b></sub></a><br /></td>
    <td align="center"><a href="https://github.com/emrecncelik"><img src="https://avatars.githubusercontent.com/u/20845117?v=4" width="100px;" alt=""/><br /><sub><b>Emrecan Çelik</b></sub></a><br /></td>
  </tr>
</table>