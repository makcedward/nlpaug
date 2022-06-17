import gzip
import os
import shutil
import tarfile
import urllib
import zipfile

import gdown
import requests


class DownloadUtil:
    """
    Helper function for downloading external dependency

    >>> from nlpaug.util.file.download import DownloadUtil
    """

    @staticmethod
    def download_word2vec(dest_dir: str = "."):
        """
        :param str dest_dir: Directory of saving file
        :return: Word2Vec C binary file named 'GoogleNews-vectors-negative300.bin'

        >>> DownloadUtil.download_word2vec('.')

        """
        file_path = DownloadUtil.download_from_google_drive(
            url="https://drive.google.com/uc?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM",
            dest_dir=dest_dir,
            dest_file="GoogleNews-vectors-negative300.bin.gz",
        )
        DownloadUtil.unzip(file_path, dest_dir=dest_dir)

    @staticmethod
    def download_glove(model_name, dest_dir):
        """
        :param str model_name: GloVe pre-trained model name. Possible values are 'glove.6B', 'glove.42B.300d',
            'glove.840B.300d' and 'glove.twitter.27B'
        :param str dest_dir: Directory of saving file

        >>> DownloadUtil.download_glove('glove.6B', '.')

        """

        url = ""
        if model_name == "glove.6B":
            url = "http://nlp.stanford.edu/data/glove.6B.zip"
        elif model_name == "glove.42B.300d":
            url = "http://nlp.stanford.edu/data/glove.42B.300d.zip"
        elif model_name == "glove.840B.300d":
            url = "http://nlp.stanford.edu/data/glove.840B.300d.zip"
        elif model_name == "glove.twitter.27B":
            url = ("http://nlp.stanford.edu/data/glove.twitter.27B.zip",)
        else:
            possible_values = [
                "glove.6B",
                "glove.42B.300d",
                "glove.840B.300d",
                "glove.twitter.27B",
            ]
            raise ValueError(
                "Unknown model_name. Possible values are {}".format(possible_values)
            )

        file_path = DownloadUtil.download(url, dest_dir=dest_dir)
        DownloadUtil.unzip(file_path)

    @staticmethod
    def download_fasttext(model_name, dest_dir):
        """
        :param str model_name: GloVe pre-trained model name. Possible values are 'wiki-news-300d-1M',
            'wiki-news-300d-1M-subword', 'crawl-300d-2M' and 'crawl-300d-2M-subword'
        :param str dest_dir: Directory of saving file

        >>> DownloadUtil.download_fasttext('glove.6B', '.')

        """

        url = ""
        if model_name == "wiki-news-300d-1M":
            url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip"
        elif model_name == "wiki-news-300d-1M-subword":
            url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip"
        elif model_name == "crawl-300d-2M":
            url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip"
        elif model_name == "crawl-300d-2M-subword":
            url = "https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip"
        else:
            possible_values = ["wiki-news-300d-1M", "crawl-300d-2M"]
            raise ValueError(
                "Unknown model_name. Possible values are {}".format(possible_values)
            )

        file_path = DownloadUtil.download(url, dest_dir=dest_dir)
        DownloadUtil.unzip(file_path)

    @staticmethod
    def download_back_translation(dest_dir):
        url = "https://storage.googleapis.com/uda_model/text/back_trans_checkpoints.zip"
        file_path = DownloadUtil.download(url, dest_dir=dest_dir)
        DownloadUtil.unzip(file_path)

    @staticmethod
    def download(src, dest_dir, dest_file=None):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        if dest_file is None:
            dest_file = os.path.basename(src)

        if not os.path.exists(dest_dir + dest_file):
            req = urllib.request.Request(src)
            file = urllib.request.urlopen(req)
            with open(os.path.join(dest_dir, dest_file), "wb") as output:
                output.write(file.read())
        return os.path.join(dest_dir, dest_file)

    @staticmethod
    def unzip(file_path, dest_dir=None):
        """
        :param str file_path: File path for unzip

        >>> DownloadUtil.unzip('zip_file.zip')

        """

        if dest_dir is None:
            dest_dir = os.path.dirname(file_path)

        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(dest_dir)
        elif file_path.endswith("tar.gz") or file_path.endswith("tgz"):
            tar = tarfile.open(file_path, "r:gz")
            tar.extractall(dest_dir)
            tar.close()
        elif file_path.endswith("tar"):
            tar = tarfile.open(file_path, "r:")
            tar.extractall(dest_dir)
            tar.close()
        elif file_path.endswith("bin.gz"):
            with gzip.open(file_path, "rb") as f_in:
                with open(file_path.replace(".gz", ""), "wb") as f_out:
                    shutil.copyfileobj(f_in, f_out)

    @staticmethod
    def download_from_google_drive(
        url: str = "https://drive.google.com/uc?export=download&id=0B7XkCwpI5KDYNlNUTTlSS21pQmM",
        dest_dir: str = ".",
        dest_file: str = "/tmp/nlpaug_model.zip",
    ) -> str:
        return gdown.download(url, output=f"{dest_dir}/{dest_file}", quiet=False)
