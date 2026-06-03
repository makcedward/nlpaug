import os
from pathlib import Path
import importlib.util

import pytest


ROOT = Path(__file__).resolve().parents[1]
TEST_RES_DIR = ROOT / "test" / "res"
PACKAGE_ROOT = ROOT / "nlpaug"
INTEGRATION_PATH_SNIPPETS = (
    "test/augmenter/test_text_augmenter.py",
    "test/augmenter/word/test_synonym.py",
    "test/augmenter/word/test_antonym.py",
    "test/augmenter/word/test_word.py",
    "test/augmenter/word/test_word_embs.py",
    "test/augmenter/word/test_context_word_embs.py",
    "test/augmenter/word/test_back_translation.py",
    "test/augmenter/word/test_tfidf.py",
    "test/augmenter/sentence/test_abst_summ.py",
    "test/augmenter/sentence/test_context_word_embs_sentence.py",
    "test/augmenter/sentence/test_lambada.py",
    "test/model/word/test_word_embs_model.py",
    "test/profiling/",
)


def pytest_configure(config):
    os.environ.setdefault("TEST_DIR", str(ROOT / "test"))
    os.environ.setdefault("PACKAGE_DIR", str(PACKAGE_ROOT))
    os.environ.setdefault("MODEL_DIR", str(ROOT / ".pytest_models"))
    config.addinivalue_line(
        "markers",
        "integration: optional tests that require external models, corpora, or heavyweight optional dependencies",
    )


def pytest_collection_modifyitems(config, items):
    integration_tests = (
        "test/flow/test_flow.py::TestFlow::test_multi_thread",
    )
    has_librosa = importlib.util.find_spec("librosa") is not None
    has_nltk = importlib.util.find_spec("nltk") is not None
    has_torch = importlib.util.find_spec("torch") is not None
    marker_expr = (config.getoption("-m") or "").strip()

    if marker_expr != "integration":
        items[:] = [item for item in items if item.nodeid not in integration_tests]

    for item in items:
        if item.nodeid in integration_tests or any(node in item.nodeid for node in INTEGRATION_PATH_SNIPPETS):
            item.add_marker(pytest.mark.integration)
        if not has_librosa and (
            "test/augmenter/audio/" in item.nodeid
            or "test/augmenter/spectrogram/" in item.nodeid
            or item.nodeid in {
                "test/augmenter/test_audio_augmenter.py::TestAudioAugmenter::test_augmenter_n_output",
                "test/augmenter/test_audio_augmenter.py::TestAudioAugmenter::test_augmenter_n_output_thread",
                "test/flow/test_flow.py::TestFlow::test_n_output_audio",
                "test/flow/test_flow.py::TestFlow::test_n_output_spectrogram",
                "test/flow/test_sequential.py::TestSequential::test_audio",
                "test/flow/test_sequential.py::TestSequential::test_spectrogram",
            }
        ):
            item.add_marker(pytest.mark.skip(reason="audio tests require the optional librosa extra"))
        if not has_torch and "test/util/selection/test_filtering.py" in item.nodeid:
            item.add_marker(pytest.mark.skip(reason="torch-backed filtering checks require the optional torch extra"))
        if not has_nltk and "test/augmenter/sentence/test_random.py" in item.nodeid:
            item.add_marker(pytest.mark.skip(reason="sentence randomization tests require the optional nltk extra"))


def pytest_ignore_collect(collection_path, config):
    marker_expr = (config.getoption("-m") or "").strip()
    if marker_expr == "integration":
        return False
    path = str(collection_path)
    return any(snippet in path for snippet in INTEGRATION_PATH_SNIPPETS)


@pytest.fixture(scope="session")
def sample_audio_path():
    return TEST_RES_DIR / "audio" / "Yamaha-V50-Rock-Beat-120bpm.wav"
