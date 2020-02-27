from setuptools import setup, find_packages
import sys

if sys.version_info < (3,):
    sys.exit("Sorry, Python3 is required.")

with open("README.md", encoding="utf8") as f:
    readme = f.read()

setup(
    name="nlpaug",
    version="0.0.13",
    author="Edward Ma",
    author_email="makcedward@gmail.com",
    url="https://github.com/makcedward/nlpaug",
    license="MIT",
    description="Natural language processing augmentation library for deep neural networks",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude="test"),
    keywords=[
        "deep learning", "neural network", "machine learning",
        "nlp", "natural language processing", "text", "audio", "spectrogram",
        "augmentation", "adversarial attack", "ai", "ml"]
)
