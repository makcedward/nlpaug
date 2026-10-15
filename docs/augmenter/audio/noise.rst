nlpaug.augmenter.audio\.noise
==============================================

``NoiseAug`` adds either generated colored noise or a supplied background-noise sample.
Set ``snr_db`` to control its strength using the RMS signal-to-noise ratio over the
augmented segment. For example, ``snr_db=20`` scales the injected noise to 20 dB below
the segment signal. Higher values produce quieter noise. The default ``None`` retains
the source noise amplitude and does not enforce an SNR.

.. code-block:: python

    import nlpaug.augmenter.audio as naa

    aug = naa.NoiseAug(snr_db=20)
    augmented_audio = aug.augment(audio)[0]

.. automodule:: nlpaug.augmenter.audio.noise
    :members:
    :inherited-members:
    :show-inheritance:
