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
| [Reference](https://github.com/makcedward/nlpaug#reference) | Refernce of external resources such as data or model |

## Quick Demo
*   [Quick Example](https://github.com/makcedward/nlpaug/blob/master/example/quick_example.ipynb)
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
|Textual| | ReservedAug | substitute | Replace reserved words |
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

If you use ContextualWordEmbsAug, ContextualWordEmbsForSentenceAug and AbstSummAug, installing the following dependencies as well
```bash
pip install torch>=1.6.0 transformers>=4.0.0
```

If you use BackTranslationAug, have to use python either 3.7 or 3.8. Also, installing the following dependencies as well
```bash
pip install torch>=1.6.0 fairseq>=0.9.0 sacremoses>=0.0.43 fastBPE>=0.1.0
```

If you use AntonymAug, SynonymAug, installing the following dependencies as well
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

If you use PitchAug, SpeedAug and VtlpAug, installing the following dependencies as well
```bash
pip install librosa>=0.7.1 matplotlib
```

## Recent Changes

### 1.1.2dev, Dec, 2020
*   Add NormalizeAug (audio)

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
*   A. Ollagnier and H. Williams. [Text Augmentation Techniques for Clinical Case Classification](https://www.researchgate.net/profile/Ollagnier_Anais/publication/343949092_Text_Augmentation_Techniques_for_Clinical_Case_Classification/links/5f49602b458515a88b810e4a/Text-Augmentation-Techniques-for-Clinical-Case-Classification.pdf). 2020
*   V. Atliha and D. Šešok. [Text Augmentation Using BERT for Image Captioning](https://www.mdpi.com/2076-3417/10/17/5978/pdf). 2020
*   Y. Ma, X. Xu, and Y. Li. [LungRN+NL: An Improved Adventitious Lung Sound Classification Using non-local block ResNet Neural Network with Mixup Data Augmentation](https://www.researchgate.net/profile/Yi_Ma5/publication/343524153_LungRNNL_An_Improved_Adventitious_Lung_Sound_Classification_Using_non-local_block_ResNet_Neural_Network_with_Mixup_Data_Augmentation/links/5f2e6158458515b7290d454d/LungRN-NL-An-Improved-Adventitious-Lung-Sound-Classification-Using-non-local-block-ResNet-Neural-Network-with-Mixup-Data-Augmentation.pdf). 2020
*   S. N. Zisad, M. Shahadat and K. Andersson. [Speech emotion recognition in neurological disorders using Convolutional Neural Network](http://www.diva-portal.org/smash/get/diva2:1456134/FULLTEXT01.pdf). 2020
*   M. Bhange and N. Kasliwal. [HinglishNLP: Fine-tuned Language Models for Hinglish Sentiment Detection](https://arxiv.org/pdf/2008.09820.pdf). 2020
*   T. Deruyttere, S. Vandenhende, D. Grujicic, Y. Liu, L. V. Gool, M. Blaschko, T. v and M. Moens. [Commands 4 Autonomous Vehicles (C4AV) Workshop Summary](https://arxiv.org/pdf/2009.08792.pdf). 2020
*   A. Tamkin, M. Wu and N. Goodman. [Viewmaker Networks: Learning Views for Unsupervised Representation Learning](https://arxiv.org/pdf/2010.07432.pdf). 2020
*   A. Spiegel, V. Cheong, J E. Kaplan and A. Sanchez. [MK-SQUIT: Synthesizing Questions using Iterative Template-Filling](https://arxiv.org/pdf/2011.02566.pdf). 2020
*   C. Zuo, N. Acharya and R. Banerjee. [Querying Across Genres for Medical Claims in News](https://www.aclweb.org/anthology/2020.emnlp-main.139.pdf). 2020
*   A. Sengupta. [DATAMAFIA at WNUT-2020 Task 2: A Study of Pre-trained Language Models along with Regularization Techniques for Downstream Tasks](https://www.aclweb.org/anthology/2020.wnut-1.51.pdf). 2020
*   V. Awatramani and A. Kumar. [Linguist Geeks on WNUT-2020 Task 2: COVID-19 Informative Tweet Identification using Progressive Trained Language Models and Data Augmentation](https://www.aclweb.org/anthology/2020.wnut-1.59.pdf). 2020
*   S. Gerani1, R. Tissot, A Ying, J. Redmon, A. Rimando and R. Hun. [Reducing suicide contagion effect by detecting sentences from media reports with explicit methods of suicide](https://crcs.seas.harvard.edu/files/crcs/files/ai4sg-21_paper_39.pdf). 2020
*   B. Velichkov, S. Gerginov, P. Panayotov, S. Vassileva, G. Velchev, I. Koyche and  S. Boytcheva. Automatic ICD-10 codes association to diagnosis: Bulgarian case. 2020
*   T. Li, X. Chen, S. Zhang, Z. Dong and K. Keutzer. [Cross-Domain Sentiment Classification with In-Domain Contrastive Learning](https://arxiv.org/pdf/2012.02943.pdf). 2020

## Project cited nlpaug
*   D. Garcia-Olano and A. Jain. [Generating Counterfactual Explanations using Reinforcement Learning Methods for Tabular and Text data](http://www.diegoolano.com/files/RL_course_Fall_2019_Final_Project.pdf). 2019
*   L. Yi. [Avengers: Achieving Superhuman Performance for Question Answering on SQuAD 2.0 Using Multiple Data Augmentations, Randomized Mini-Batch Training and Architecture Ensembling](https://pdfs.semanticscholar.org/ce36/6e8f69a26ea84a65fc2b37d7492f6c8993fe.pdf). 2020

## Contributions (Supporting Other Languages)
- [sakares](https://github.com/sakares): Add Thai support to KeyboardAug
