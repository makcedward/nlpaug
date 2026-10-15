Common Augmenter Parameters
===========================

Augmenters expose only the parameters that apply to their operation. The class-specific
API pages list the exact constructor signature; this page explains the options shared by
character and word augmenters.

Augmentation amount
-------------------

``aug_p``
    Fraction of eligible words to augment. ``0.3`` means approximately 30 percent.

``aug_min`` / ``aug_max``
    Lower and upper bounds for the number of words to augment. The calculated ``aug_p``
    count is constrained by these bounds. ``None`` disables the corresponding bound.

``aug_char_p``
    Fraction of eligible characters to augment inside each selected word.

``aug_char_min`` / ``aug_char_max``
    Lower and upper bounds for the number of characters changed in each selected word.

``aug_word_p``
    Fraction of eligible words selected by a character augmenter. This controls which
    words are considered before ``aug_char_p`` controls characters within those words.

``aug_word_min`` / ``aug_word_max``
    Lower and upper bounds for the number of words selected by a character augmenter.

Eligibility and filtering
-------------------------

``min_char``
    Skip words shorter than this number of characters.

``stopwords``
    Exact words or multi-token phrases that must not be augmented.

``stopwords_regex``
    Regular expression used to exclude matching words. Use this when the exclusion rule
    is pattern-based rather than a fixed list.

``include_upper_case`` / ``include_lower_case``
    Allow generated character candidates from the corresponding letter case.

``include_numeric`` / ``include_special_char``
    Allow generated numeric or punctuation candidates. Availability depends on the
    augmenter and its model.

Tokenization and output
-----------------------

``tokenizer``
    Callable that accepts one string and returns an ordered list of tokens. Supplying a
    custom tokenizer normally requires a matching ``reverse_tokenizer``.

``reverse_tokenizer``
    Callable that accepts the augmented token list and reconstructs a string. The default
    tokenizer pair preserves punctuation and the original separators where supported.

``include_detail``
    Return each augmented string together with token-level change logs. See
    :doc:`change_log` for field definitions and position semantics.

``n``
    Argument to ``augment(...)`` specifying the requested number of unique outputs for a
    single input. A list input returns one result per input item.

``num_thread``
    Argument to ``augment(...)`` controlling CPU worker threads. Keep it at ``1`` for GPU
    inference and increase it only after measuring CPU-bound workloads.

``verbose``
    Controls diagnostic output. The default ``0`` is quiet.

Model-backed augmenters
-----------------------

``device``
    Inference device such as ``"cpu"``, ``"cuda"``, or ``"cuda:1"``. PyTorch device
    objects are also accepted by language-model wrappers.

``batch_size``
    Maximum number of inputs processed together by supported model-backed augmenters.

``force_reload``
    Bypass the process-wide model cache and load a fresh model instance. Leave this
    disabled for normal use to avoid repeated initialization.

``top_k`` / ``top_p`` / ``temperature``
    Candidate sampling controls for supported language-model augmenters. Smaller pools
    are more conservative; temperature controls sampling sharpness.

Example
-------

.. code-block:: python

    import nlpaug.augmenter.char as nac

    aug = nac.KeyboardAug(
        aug_word_p=0.2,
        aug_word_min=1,
        aug_char_p=0.1,
        aug_char_min=1,
        min_char=4,
        stopwords=["nlpaug"],
    )
    augmented = aug.augment("nlpaug makes text augmentation straightforward", n=3)

See the individual augmenter API page for defaults and options unique to that class.
