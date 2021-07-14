Source
======

Pre-trained Model File
----------------------
*  [word2vec](https://code.google.com/archive/p/word2vec/) (Google): Tomas Mikolov, Kai Chen, Greg Corrado and Jeffrey Dean released [Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/pdf/1301.3781.pdf)
*  [GloVe](https://nlp.stanford.edu/projects/glove/) (Standford): Jeffrey Pennington, Richard Socher, and Christopher D. Manning released [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf)
*  [fastText](https://fasttext.cc/docs/en/english-vectors.html) (Facebook): Tomas Mikolov, Edouard Grave, Piotr Bojanowski, Christian Puhrsch and Armand Joulin released [Advances in Pre-Training Distributed Word Representations](https://arxiv.org/pdf/1712.09405.pdf)
*  [BERT](https://github.com/google-research/bert) (Google): Jacob Devlin, Ming-Wei Chang, Kenton Lee and Kristina Toutanova released [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). Used [Hugging Face](https://huggingface.co/) [PyTorch version](https://github.com/huggingface/transformers).
*  [RoBERTa](https://github.com/pytorch/fairseq) (UW/Facebook): Y. Liu, M. Ott, N. Goyal, J. Du, M. Joshi, D. Chen, O. Levy, M. Lewis, L. Zettlemoyer, and V. Stoyanov released [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://medium.com/towards-artificial-intelligence/a-robustly-optimized-bert-pretraining-approach-f6b6e537e6a6). Used [Hugging Face](https://huggingface.co/) [PyTorch version](https://github.com/huggingface/transformers).
*  [DistilBERT](https://github.com/huggingface/transformers) (Hugging Face): . Used [Hugging Face](https://huggingface.co/) [PyTorch version](https://github.com/huggingface/transformers).
*  [GPT2](https://github.com/openai/gpt-2) (OpenAI): Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei and Ilya Sutskever released [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). Used [Hugging Face](https://huggingface.co/) [PyTorch version](https://github.com/huggingface/transformers).
*  [DistilGPT2](https://github.com/huggingface/transformers) (Hugging Face): Used [Hugging Face](https://huggingface.co/) [PyTorch version](https://github.com/huggingface/transformers).
*  [XLNet](https://github.com/zihangdai/xlnet) (Google/CMU): Zhilin Yang, Zihang Dai, Yiming Yang, Jaime Carbonell, Ruslan Salakhutdinov, Quoc V. Le released [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237). Used [Hugging Face](https://huggingface.co/) [PyTorch version](https://github.com/huggingface/transformers).
*  [Fairseq WMT19](https://github.com/pytorch/fairseq) (Facebook): Nathan Ng, Kyra Yee, Alexei Baevski, Myle Ott, Michael Auli and Sergey Edunov released [Facebook FAIR’s WMT19 News Translation Task Submission](https://arxiv.org/pdf/1907.06616.pdf)


Raw Data Source
---------------
*   data/Yamaha-V50-Rock-Beat-120bpm.wav, [source](https://freewavesamples.com/yamaha-v50-rock-beat-120-bpm)
*   model/spelling_en.txt, [source](https://github.com/ybisk/charNMT-noise)

Research Reference
------------------
Some of the above augmenters are inspired by the following research papers. However, it does not always follow original implementation due to different reasons. If original implementation is needed, please refer to original source code.

*   J. Salamon and J. P. Bello. [Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification](https://arxiv.org/pdf/1608.04363.pdf). 2016
*   Y. Belinkov and Y. Bisk. [Synthetic and Natural Noise Both Break Neural Machine Translation](https://arxiv.org/pdf/1711.02173.pdf). 2017
*   J. Ebrahimi, A. Rao, D. Lowd and D. Dou. [HotFlip: White-Box Adversarial Examples for Text Classification](https://arxiv.org/pdf/1712.06751.pdf). 2018
*   J. Ebrahimi, D. Lowd and Dou. [On Adversarial Examples for Character-Level Neural Machine Translation](https://arxiv.org/pdf/1806.09030.pdf). 2018
*   D. Pruthi, B. Dhingra and Z. C. Lipton. [Combating Adversarial Misspellings with Robust Word Recognition](https://arxiv.org/pdf/1905.11268.pdf). 2019
*   T. Niu and M. Bansal. [Adversarial Over-Sensitivity and Over-Stability Strategies for Dialogue Models](https://arxiv.org/pdf/1809.02079.pdf). 2018
*   P. Minervini and S. Riedel. [Adversarially Regularising Neural NLI Models to Integrate Logical Background Knowledge](https://arxiv.org/pdf/1808.08609.pdf). 2018
*   X. Zhang, J. Zhao and Y. LeCun. [Character-level Convolutional Networks for Text Classification](https://arxiv.org/pdf/1509.01626.pdf). 2015
*   S. Kobayashi and C. Coulombe. [Text Data Augmentation Made Simple By Leveraging NLP Cloud APIs](https://arxiv.org/ftp/arxiv/papers/1812/1812.04718.pdf). 2018
*   Q. Xie, Z. Dai, E Hovy, M. T. Luong and Q. V. Le. [Unsupervised Data Augmentation](https://arxiv.org/pdf/1904.12848.pdf). 2019
*   W. Y. Wang and D. Yang. [That’s So Annoying!!!: A Lexical and Frame-Semantic Embedding Based Data Augmentation Approach to Automatic Categorization of Annoying Behaviors using #petpeeve Tweets](https://aclweb.org/anthology/D15-1306). 2015
*   S. Kobayashi. [Contextual Augmentation: Data Augmentation by Words with Paradigmatic Relation](https://arxiv.org/pdf/1805.06201.pdf). 2018
*   D. S. Park, W. Chan, Y. Zhang, C. C. Chiu, B. Zoph, E. D. Cubuk and Q. V. Le. [SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition](https://arxiv.org/pdf/1904.08779.pdf). 2019
*   R. Jia and P. Liang. [Adversarial Examples for Evaluating Reading Comprehension Systems](https://arxiv.org/pdf/1707.07328.pdf). 2017
*   M. Alzantot, Y. Sharma, A. Elgohary, B. Ho, M. B. Srivastava and K. Chang. [Generating Natural Language Adversarial Examples](https://arxiv.org/pdf/1804.07998.pdf). 2018
*   Z. Xie, S. I. Wang, J. Li, D. Levy, A. Nie, D. Jurafsky and A. Y. Ng. [Data Noising as Smoothing in Natural Network Language Models](https://arxiv.org/pdf/1703.02573.pdf). 2017
*   N. Jaitly and G. E. Hinton. [Vocal Tract Length Perturbation (VTLP) improves speech recognition](https://pdfs.semanticscholar.org/3de0/616eb3cd4554fdf9fd65c9c82f2605a17413.pdf). 2013
*	N. Ng, K. Yee, A. Baevski, M. Ott, M. Auli and S Edunov. [Facebook FAIR’s WMT19 News Translation Task Submission](https://arxiv.org/pdf/1907.06616.pdf). 2019
*	V. Kumar, A. Choudhary and E. Cho. [Data Augmentation using Pre-trained Transformer Models](https://arxiv.org/pdf/2003.02245.pdf). 2020
*	Y. Hwang, H. Cho, H. Yang, D. Won, I. Oh and S. Lee. [Mel-spectrogram augmentation for sequence-to-sequence voice conversion](https://arxiv.org/pdf/2001.01401.pdf). 2020
*	G. G. Sahin and M. Steedman. [Data Augmentation via Dependency Tree Morphing for Low-Resource Languages](https://arxiv.org/pdf/1903.09460.pdf). 2019
*	M. Regina, M. Meyer and S. Goutal [Text Data Augmentation: Towards better detection of spear-phishing emails](https://arxiv.org/pdf/2007.02033.pdf). 2020
*	A. Anaby-Tavor, B. Carmeli, E. Goldbraich, A. Kantor, G. Kour, S. Shlomov, N. Tepper, N. Zwerdling. [Do Not Have Enough Data? Deep Learning to the Rescue!
](https://arxiv.org/pdf/1911.03118.pdf). 2019