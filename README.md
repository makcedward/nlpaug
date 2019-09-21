[![Build Status](https://travis-ci.org/makcedward/nlpaug.svg?branch=master)](https://travis-ci.org/makcedward/nlpaug)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/2d6d1d08016a4f78818161a89a2dfbfb)](https://www.codacy.com/app/makcedward/nlpaug?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=makcedward/nlpaug&amp;utm_campaign=Badge_Grade)

# nlpaug

This python library helps you with augmenting nlp for your machine learning projects. Visit this introduction to understand about [Data Augmentation in NLP](https://towardsdatascience.com/data-augmentation-in-nlp-2801a34dfc28). `Augmenter` is the basic element of augmentation while `Flow` is a pipeline to orchestra multi augmenter together.
*   [Data Augmentation library for Text](https://towardsdatascience.com/data-augmentation-library-for-text-9661736b13ff)
*   [Data Augmentation library for Speech Recognition](https://towardsdatascience.com/data-augmentation-for-speech-recognition-e7c607482e78)
*   [Data Augmentation library for Audio](https://towardsdatascience.com/data-augmentation-for-audio-76912b01fdf6)
*   [Does your NLP model able to prevent adversarial attack?](https://medium.com/hackernoon/does-your-nlp-model-able-to-prevent-adversarial-attack-45b5ab75129c)

## Starter Guides
*   [Example of Augmentation for Textual Inputs](https://github.com/makcedward/nlpaug/blob/master/example/textual_augmenter.ipynb)
*   [Example of Augmentation for Spectrogram Inputs](https://github.com/makcedward/nlpaug/blob/master/example/spectrogram_augmenter.ipynb)
*   [Example of Augmentation for Audio Inputs](https://github.com/makcedward/nlpaug/blob/master/example/audio_augmenter.ipynb)
*   [Example of Orchestra Multiple Augmenters](https://github.com/makcedward/nlpaug/blob/master/example/flow.ipynb)
*   How to train [TF-IDF model](https://github.com/makcedward/nlpaug/blob/master/example/tfidf-train_model.ipynb)
*   How to create [custom augmentation](https://github.com/makcedward/nlpaug/blob/master/example/custom_augmenter.ipynb)
*   [API Documentation](https://nlpaug.readthedocs.io/en/latest/)

## Flow
| Pipeline | Description |
|:---:|:---:|
| Sequential | Apply list of augmentation functions sequentially |
| Sometimes | Apply some augmentation functions randomly |


## Textual Augmenter
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
| | SynonymAug | substitute | Substitute similar word according to WordNet/ PPDB synonym |
| | AntonymAug | substitute | Substitute opposite meaning word according to WordNet antonym|
| | WordEmbsAug | insert | Insert word randomly from [word2vec](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a), [GloVe](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a) or [fasttext](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a) dictionary |
| | | substitute | Substitute word based on [word2vec](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a), [GloVe](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a) or [fasttext](https://towardsdatascience.com/3-silver-bullets-of-word-embedding-in-nlp-10fa8f50cc5a) embeddings |
| | TfIdfAug | insert | Insert word randomly trained TF-IDF model |
| | | substitute | Substitute word based on TF-IDF score |
| | ContextualWordEmbsAug | insert | Insert word based by feeding surroundings word to [BERT](https://towardsdatascience.com/how-bert-leverage-attention-mechanism-and-transformer-to-learn-word-contextual-relations-5bbee1b6dbdb) and XLNet language model |
| | | substitute | Substitute word based by feeding surroundings word to [BERT](https://towardsdatascience.com/how-bert-leverage-attention-mechanism-and-transformer-to-learn-word-contextual-relations-5bbee1b6dbdb) and XLNet language model |
| Sentence | ContextualWordEmbsForSentenceAug | insert | Insert sentence according to GPT2 or XLNet prediction |

## Signal Augmenter
| Target | Augmenter | Action | Description |
|:---:|:---:|:---:|:---:|
| Audio | NoiseAug | substitute | Inject noise |
| | PitchAug | substitute | Adjust audio's pitch |
| | ShiftAug | substitute | Shift time dimension forward/ backward |
| | SpeedAug | substitute | Adjust audio's speed |
| | CropAug | delete | Delete audio's segment |
| | LoudnessAug|substitute | Adjust audio's volume |
| | MaskAug | substitute | Mask audio's segment |
| Spectrogram | FrequencyMaskingAug | substitute | Set block of values to zero according to frequency dimension |
| | TimeMaskingAug | substitute | Set block of values to zero according to time dimension |

## Installation

The library supports python 3.5+ in linux and window platform.

To install the library:
```bash
pip install nlpaug numpy matplotlib python-dotenv
```
or install the latest version (include BETA features) from github directly
```bash
pip install git+https://github.com/makcedward/nlpaug.git
```

If you use ContextualWordEmbsAug or ContextualWordEmbsForSentenceAug, install the following dependencies as well
```bash
pip install torch>=1.1.0 pytorch_pretrained_bert>=1.1.0
```

If you use AntonymAug, SynonymAug, install the following dependencies as well
```bash
pip install nltk
```

If you use WordEmbsAug (word2vec, glove or fasttext), downloading pre-trained model first
```bash
from nlpaug.util.file.download import DownloadUtil
DownloadUtil.download_word2vec(dest_dir='.') # Download word2vec model
DownloadUtil.download_glove(model_name='glove.6B', dest_dir='.') # Download GloVe model
DownloadUtil.download_fasttext(model_name='wiki-news-300d-1M', dest_dir='.') # Download fasttext model
```

If you use any one of audio augmenter, install the following dependencies as well
```bash
pip install librosa
```

## Recent Changes

**BETA** Sep, 2019
*   Added Swap Mode (adjacent, middle and random) for RandomAug (character level)
*   Added SynonymAug (WordNet/ PPDB) and AntonymAug (WordNet)
*   WordNetAug is deprecated. Uses SynonymAug instead
*   Introduce parameter n. Returning more than 1 augmented data. Changing output format from text (or numpy) to list of text (or numpy) if n > 1

**0.0.8** Sep 4, 2019
*   BertAug is replaced by ContextualWordEmbsAug
*   Support GPU (for ContextualWordEmbsAug and ContextualWordEmbsForSentenceAug only) [#26](https://github.com/makcedward/nlpaug/issues/26)
*   Upgraded pytorch_transformer to 1.1.0 version [#33](https://github.com/makcedward/nlpaug/issues/33)
*   ContextualWordEmbsAug suuports both BERT and XLNet model
*   Removed librosa dependency
*   Add ContextualWordEmbsForSentenceAug for generating next sentence
*   Fix sampling issue [#38](https://github.com/makcedward/nlpaug/issues/38)

See [changelog](https://github.com/makcedward/nlpaug/blob/master/CHANGE.md) for more details.

## Source
This library uses data (e.g. capturing from internet), research (e.g. following augmenter idea), model (e.g. using pre-trained model) See [data source](https://github.com/makcedward/nlpaug/blob/master/SOURCE.md) for more details.
