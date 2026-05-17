Release Note
^^^^^^^^^^^^

0.4.0 - 13 Oct 2025
-------------------

* Add Keras 3 / TensorFlow 2.16-2.20 compatibility fixes.
* Update the tested dependency floor to Python 3.9, NumPy 1.26+, and librosa 0.11+.
* Add type-checking configuration and refresh serialization tests.

0.3.7 - 21 Jan 2022
-------------------

* Add `SpecAugment <https://github.com/keunwoochoi/kapre/pull/135>`_ layer.

0.3.6 - 13 Nov 2021
-------------------

* Fix ``pad_end`` in the TFLite-compatible layers, via #131.

0.3.5 - 18 March 2021
---------------------

* Add ``kapre.time_frequency_tflite`` which uses TFLite for faster CPU inference.

0.3.4 - 29 Sep 2020
-------------------

* Fix a bug in ``kapre.backend.get_window_fn()``. Previously, it only correctly worked with ``None`` input and an error was raised when non-default value was set for ``window_name`` in any layer.

0.3.3 - 15 Sep 2020
-------------------

* Add ``kapre.augmentation``.
* Add ``kapre.time_frequency.ConcatenateFrequencyMap``.
* Add ``kapre.composed.get_frequency_aware_conv2d``.
* In ``STFT`` and ``InverseSTFT``, rename keyword argument ``window_fn`` to ``window_name`` and expect a string value, not a function. With this update, models with Kapre layers can be loaded with ``h5`` file format.
* Add ``kapre.backend.get_window_fn``.

0.3.2 - 28 Aug 2020
-------------------

* Add ``kapre.signal.Frame`` and ``kapre.signal.Energy``.
* Add ``kapre.signal.LogmelToMFCC``.
* Add ``kapre.signal.MuLawEncoder`` and ``kapre.signal.MuLawDecoder``.
* Add ``kapre.composed.get_stft_magnitude_layer()``.

0.3.1 - 21 Aug 2020
-------------------

* Add ``Inverse STFT``.

0.3.0 - 15 Aug 2020
-------------------

* Make breaking and simplifying changes with TensorFlow 2.0 and more tests. Some features are removed.

0.2.0 - 29 Jul 2020
-------------------

* Change melspectrogram filterbank from ``norm=1`` to ``norm='slaney'`` (w.r.t. Librosa) due to the update from Librosa: https://github.com/keunwoochoi/kapre/issues/77. This changes the behavior of melspectrogram slightly.
* Bump librosa version to 0.7.2 or higher.

0.1.8 - 17 Mar 2020
-------------------

* Add ``utils.Delta`` layer.

0.1.7 - 20 Feb 2020
-------------------

* Remove vanilla Keras dependency.
* Require TensorFlow >= 1.15 only.
* Stop testing on Python 2.7; test only on Python 3.6 and 3.7 locally (by ``tox``) and 3.6 on Travis.

0.1.4 - 20 Feb 2019
-------------------

* Fix amplitude-to-decibel error as raised in https://github.com/keunwoochoi/kapre/issues/46.

0.1.3 - March 2018
------------------

* Put Kapre on PyPI again.
* Add unit tests.
* Remove ``Datasets``.
* Remove some code while adding more dependency on Librosa to make it cleaner and more stable, enabling the ``htk`` option in ``Melspectrogram``.

0.1.1 - 9 July 2017
-------------------

* Release the “pretty stable” version with a benchmark paper: https://arxiv.org/abs/1706.05781.
* Remove STFT, make Python 3 compatible.
* Add full documentation in ``README.md``.
* Update the PyPI version.
