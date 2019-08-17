NLPAUG Change Log
================

**0.0.6** Jul 29, 2019:
-   Added new augmenter [TF-IDF based word replacement augmenter](https://arxiv.org/pdf/1904.12848.pdf)(TfIdfAug)
-   Added new augmenter [Spelling mistake simulation augmenter](https://arxiv.org/pdf/1711.02173.pdf)(SpellingAug)
-   Added new augmenter [Stopword Dropout augmenter](https://arxiv.org/pdf/1809.02079.pdf)(StopWordsAug)
-   Fixed [#14](https://github.com/makcedward/nlpaug/issues/14)

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