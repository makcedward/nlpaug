from setuptools import setup, find_packages
import sys

if sys.version_info < (3,):
    sys.exit("Sorry, Python3 is required.")

with open("README.md", encoding="utf8") as f:
    readme = f.read()

with open('requirements.txt') as f:
    install_reqs = f.read().splitlines()

setup(
    name="nlpaug",
    version="1.1.11",
    author="Edward Ma",
    author_email="makcedward@gmail.com",
    url="https://github.com/makcedward/nlpaug",
    license="MIT",
    description="Natural language processing augmentation library for deep neural networks",
    long_description=readme,
    long_description_content_type="text/markdown",
    packages=find_packages(exclude="test"),
    include_package_data=True,
    install_requires=install_reqs,
    keywords=[
        "deep learning", "neural network", "machine learning",
        "nlp", "natural language processing", "text", "audio", "spectrogram",
        "augmentation", "adversarial attack", "ai", "ml"],
    python_requires=">=3.7"
)
