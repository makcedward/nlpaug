nlpaug.augmenter.word\.context_word_embs
========================================

``ContextualWordEmbsAug`` supports BERT, DistilBERT, RoBERTa, and BART masked-language
models. BART checkpoints use the Hugging Face fill-mask backend automatically, including
when ``use_custom_api`` retains its default value.

.. code-block:: python

    import nlpaug.augmenter.word as naw

    aug = naw.ContextualWordEmbsAug(
        model_path="facebook/bart-base",
        action="substitute",
    )
    augmented_text = aug.augment("The quick brown fox.")[0]

.. automodule:: nlpaug.augmenter.word.context_word_embs
    :members:
    :inherited-members:
    :show-inheritance:
