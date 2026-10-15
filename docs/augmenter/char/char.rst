Character Augmenter
===================

With the default tokenizer, character augmenters preserve the input's original
separators. This includes leading and trailing whitespace, repeated spaces, and spaces
around punctuation. Supplying a custom ``tokenizer`` also requires a matching
``reverse_tokenizer`` if the custom tokenization needs the same preservation behavior.

.. toctree::
    :maxdepth: 6

    ./keyboard
    ./ocr
    ./random
