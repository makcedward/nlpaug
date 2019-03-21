import os, urllib, zipfile


class DownloadUtil:
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
