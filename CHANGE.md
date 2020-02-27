NLPAUG Change Log
================

**0.0.13 Feb 25, 2020
*   Fix spectrogram tutorial notebook [#98] (https://github.com/makcedward/nlpaug/issues/98)
*   Fix RandomWordAug missed aug_max parameter [#100] (https://github.com/makcedward/nlpaug/issues/100)
*   Fix loading KeyboardAug model problem [#101] (https://github.com/makcedward/nlpaug/issues/101)
*   Fix performance issue when sampling candidate in ContextualWordEmbsAug and ContextualWordEmbsForSentenceAug [#107](https://github.com/makcedward/nlpaug/issues/107)

**0.0.12 Feb 5, 2020
*   ContextualWordEmbsAug supports bert-base-multilingual-uncased (for non English inputs)
*   Fix missing library dependency [#74](https://github.com/makcedward/nlpaug/issues/74)
*   Fix single token error when using RandomWordAug [#76](https://github.com/makcedward/nlpaug/issues/76)
*   Fix replacing character in RandomCharAug error [#77](https://github.com/makcedward/nlpaug/issues/77)
*   Enhance word's augmenter to support regular expression stopwords [#81](https://github.com/makcedward/nlpaug/issues/81)
*   Enhance char's augmenter to support regular expression stopwords [#86](https://github.com/makcedward/nlpaug/issues/86)
*   KeyboardAug supports Thai language [#92](https://github.com/makcedward/nlpaug/pull/92)
*   Fix word casing issue [#82](https://github.com/makcedward/nlpaug/issues/82)

**0.0.11 Dec 6, 2019
*   Support color noise (pink, blue, red and violet noise) in audio's NoiseAug
*   Support given background noise in audio's NoiseAug
*   Support inject noise to portion of audio only in audio's NoiseAug
*   Introduce `zone`, `coverage` to all audio augmenter. Support only augmented portion of audio input
*   Add VTLP augmentation methods (Audio's augmenter)
*   Adopt latest transformer's interface [#59](https://github.com/makcedward/nlpaug/pull/59)
*   Support RoBERTa (including DistilRoBERTa) and DistilBERT (ContextualWordEmbsAug)
*   Support DistilGPT2 (ContextualWordEmbsForSentenceAug)
*   Fix librosa hard dependency [#62](https://github.com/makcedward/nlpaug/issues/62)
*   Introduce `optimize` attribute ContextualWordEmbsForSentenceAug [#63](https://github.com/makcedward/nlpaug/pull/63)
*   Optimize word selection for ContextualWordEmbsAug and ContextualWordEmbsForSentenceAug (Speed up around 30%)
*   Add retry mechanism into ContextualWordEmbsAug insert action [#68](https://github.com/makcedward/nlpaug/issues/68)

**0.0.10 Nov, 2019
*   Add aug_max to control maximum number of augmented item
*   Fix ContextualWordEmbsAug (for BERT) error when input is longer than max sequence length
*   Add RandomWordAug Substitute action
*	Fix ContextualWordEmbsAug error when no augmented data
*   Support multi thread processing (for CPU only) to speed up the augmentation
*   Fix KeyboardAug error [#55](https://github.com/makcedward/nlpaug/issues/55)

**0.0.9** Sep 30, 2019
*   Added Swap Mode (adjacent, middle and random) for RandomAug (character level)
*   Added SynonymAug (WordNet/ PPDB) and AntonymAug (WordNet)
*   WordNetAug is deprecated. Uses SynonymAug instead
*   Introduce parameter n. Returning more than 1 augmented data. Changing output format from text (or numpy) to list of text (or numpy) if n > 1
*   Introduce parameter temperature in ContextualWordEmbsAug and ContextualWordEmbsForSentenceAug to control the randomness
*   aug_n parameter is deprecated. This parameter will be replaced by top_k parameter
*   Fixed tokenization issue  [#48](https://github.com/makcedward/nlpaug/issues/48)
*   Upgraded transformers dependency (or pytorch_transformer) to 2.0.0
*   Upgraded PyTorch dependency to 1.2.0
*   Added SplitAug

**0.0.8** Sep 4, 2019
*   BertAug is replaced by ContextualWordEmbsAug
*   Support GPU (for ContextualWordEmbsAug and ContextualWordEmbsForSentenceAug only) [#26](https://github.com/makcedward/nlpaug/issues/26)
*   Upgraded pytorch_transformer to 1.1.0 version [#33](https://github.com/makcedward/nlpaug/issues/33)
*   ContextualWordEmbsAug suuports both BERT and XLNet model
*   Removed librosa dependency
*   Add ContextualWordEmbsForSentenceAug for generating next sentence
*   Fix sampling issue [#38](https://github.com/makcedward/nlpaug/issues/38)

**0.0.7** Aug 21, 2019
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

**0.0.5** Jul 2, 2019:
-   Fixed [#3](https://github.com/makcedward/nlpaug/issues/3), [#4](https://github.com/makcedward/nlpaug/issues/4), [#5](https://github.com/makcedward/nlpaug/issues/5), [#7](https://github.com/makcedward/nlpaug/issues/7), [#10](https://github.com/makcedward/nlpaug/issues/10)

**0.0.4** Jun 7, 2019:
-   Added stopwords feature in character and word augmenter.
-   Added character's swap augmenter.
-   Added word's swap augmenter.
-   Added validation rule for [#1](https://github.com/makcedward/nlpaug/issues/1).
-   Fixed BERT reverse tokenization for [#2](https://github.com/makcedward/nlpaug/issues/2).

**0.0.3** May 23, 2019:
-   Added Speed, Noise, Shift and Pitch augmenters for Audio

**0.0.2** Apr 30, 2019:
-   Added Frequency Masking and Time Masking for Speech Recognition (Spectrogram).
-   Added librosa library dependency for converting wav to spectrogram.

**0.0.1** Mar 20, 2019: Project initialization