Augmentation Change Logs
========================

Augmenters that support change logs accept ``include_detail=True``. For a single input,
``augment`` still returns a list, so unpack the first result:

.. code-block:: python

    import nlpaug.augmenter.word as naw

    aug = naw.RandomWordAug(
        action="substitute",
        target_words=["fast"],
        include_detail=True,
    )
    augmented_text, change_logs = aug.augment("The quick fox")[0]

Each change log is a dictionary with these fields:

``orig_token``
    The original tokenizer token. Model tokenizers may expose subword markers such as
    ``##`` even though those marker characters are not present in the input text.

``orig_start_pos``
    Zero-based character offset of ``orig_token`` in the original text. For a subword,
    the offset points to the first source character after removing its tokenizer marker.
    Inserted tokens use ``-1`` because they have no original position.

``new_token``
    The token after the recorded change. A deletion uses an empty string.

``new_start_pos``
    Zero-based character offset in the augmented text. For a deletion, this is the
    position where the removed token previously occurred.

``change_seq``
    One-based operation order. Multiple entries can share a sequence number when one
    operation changes several tokens, such as a swap or capitalization alignment.

``action``
    Operation that produced the entry, such as ``substitute``, ``insert``, ``delete``,
    ``swap``, ``split``, or ``align``.

Position scope
--------------

Character offsets refer to the text processed by the augmenter. Dictionary and random
augmenters normally process the supplied input directly. Transformer-backed augmenters
may normalize case, whitespace, or Unicode through their tokenizer first; their offsets
therefore refer to that normalized model input and output rather than byte offsets in the
unprocessed source. Offsets are character indexes, not UTF-8 byte indexes.
