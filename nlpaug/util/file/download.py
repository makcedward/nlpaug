import os, urllib, zipfile, requests


class DownloadUtil:
    @staticmethod
    def download_word2vec(dest_dir):
        DownloadUtil.download_from_google_drive(
            id='0B7XkCwpI5KDYNlNUTTlSS21pQmM', dest_dir=dest_dir, dest_file='GoogleNews-vectors-negative300.zip'
        )

    @staticmethod
    def download_glove(model_name, dest_dir):
        url = ''
        if model_name == 'glove.6B':
            url = 'http://nlp.stanford.edu/data/glove.6B.zip'
        elif model_name == 'glove.42B.300d':
            url = 'http://nlp.stanford.edu/data/glove.42B.300d.zip'
        elif model_name == 'glove.840B.300d':
            url = 'http://nlp.stanford.edu/data/glove.840B.300d.zip'
        elif model_name == 'glove.twitter.27B':
            url = 'http://nlp.stanford.edu/data/glove.twitter.27B.zip',
        else:
            possible_values = ['glove.6B', 'glove.42B.300d', 'glove.840B.300d', 'glove.twitter.27B']
            raise ValueError('Unknown model_name. Possible values are {}'.format(possible_values))

        file_path = DownloadUtil.download(url, dest_dir=dest_dir)
        DownloadUtil.unzip(file_path)

    @staticmethod
    def download_fasttext(model_name, dest_dir):
        url = ''
        if model_name == 'wiki-news-300d-1M':
            url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip'
        # elif model_name == 'wiki-news-300d-1M-subword':
        #     url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip'
        elif model_name == 'crawl-300d-2M':
            url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip'
        # elif model_name == 'crawl-300d-2M-subword':
        #     url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip'
        else:
            possible_values = ['wiki-news-300d-1M', 'crawl-300d-2M']
            raise ValueError('Unknown model_name. Possible values are {}'.format(possible_values))

        file_path = DownloadUtil.download(url, dest_dir=dest_dir)
        DownloadUtil.unzip(file_path)

    @staticmethod
    def download(src, dest_dir, dest_file=None):
        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        if dest_file is None:
            dest_file = os.path.basename(src)

        if not os.path.exists(dest_dir + dest_file):
            file = urllib.request.urlopen(src)
            with open(dest_dir + dest_file, 'wb') as output:
                output.write(file.read())
        return dest_dir + dest_file

    @staticmethod
    def unzip(file_path):
        dest_dir = os.path.dirname(file_path)
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(dest_dir)

    @staticmethod
    def download_from_google_drive(id, dest_dir, dest_file):
        url = "https://docs.google.com/uc?export=download"
        def get_confirm_token(response):
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value

            return None

        def save_response_content(response, destination):
            CHUNK_SIZE = 32768

            with open(destination, "wb") as f:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)

        session = requests.Session()

        response = session.get(url, params={'id': id}, stream=True)
        token = get_confirm_token(response)

        if token:
            params = {'id': id, 'confirm': token}
            response = session.get(url, params=params, stream=True)

        save_response_content(response, os.path.join(dest_dir, dest_file))
