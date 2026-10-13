import gzip
import io
import tarfile
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from nlpaug.util.file.download import DownloadUtil


def test_download_creates_destination_and_writes_file(tmp_path):
    payload = b'hello'

    class FakeResponse:
        def read(self):
            return payload

    dest_dir = tmp_path / 'downloads'
    with patch('urllib.request.urlopen', return_value=FakeResponse()), patch('urllib.request.Request', side_effect=lambda url: url):
        file_path = DownloadUtil.download('https://example.com/file.txt', str(dest_dir) + '/')

    assert Path(file_path).read_bytes() == payload


def test_download_uses_custom_destination_name(tmp_path):
    payload = b'data'

    class FakeResponse:
        def read(self):
            return payload

    with patch('urllib.request.urlopen', return_value=FakeResponse()), patch('urllib.request.Request', side_effect=lambda url: url):
        file_path = DownloadUtil.download('https://example.com/file.txt', str(tmp_path) + '/', dest_file='custom.bin')

    assert Path(file_path).name == 'custom.bin'


def test_unzip_supports_zip_tar_targz_and_gzip(tmp_path):
    zip_path = tmp_path / 'sample.zip'
    with zipfile.ZipFile(zip_path, 'w') as archive:
        archive.writestr('zip.txt', 'zip-data')
    DownloadUtil.unzip(str(zip_path), str(tmp_path / 'zip_out'))
    assert (tmp_path / 'zip_out' / 'zip.txt').read_text() == 'zip-data'

    tar_path = tmp_path / 'sample.tar'
    with tarfile.open(tar_path, 'w') as archive:
        data = b'tar-data'
        info = tarfile.TarInfo(name='tar.txt')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    DownloadUtil.unzip(str(tar_path), str(tmp_path / 'tar_out'))
    assert (tmp_path / 'tar_out' / 'tar.txt').read_text() == 'tar-data'

    targz_path = tmp_path / 'sample.tar.gz'
    with tarfile.open(targz_path, 'w:gz') as archive:
        data = b'targz-data'
        info = tarfile.TarInfo(name='targz.txt')
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    DownloadUtil.unzip(str(targz_path), str(tmp_path / 'targz_out'))
    assert (tmp_path / 'targz_out' / 'targz.txt').read_text() == 'targz-data'

    gz_path = tmp_path / 'sample.bin.gz'
    with gzip.open(gz_path, 'wb') as handle:
        handle.write(b'bin-data')
    DownloadUtil.unzip(str(gz_path))
    assert (tmp_path / 'sample.bin').read_bytes() == b'bin-data'


@pytest.mark.parametrize(
    ('archive_name', 'mode'),
    [
        ('malicious.tar', 'w'),
        ('malicious.tar.gz', 'w:gz'),
        ('malicious.tgz', 'w:gz'),
    ],
)
def test_unzip_rejects_tar_path_traversal(tmp_path, archive_name, mode):
    archive_path = tmp_path / archive_name
    extraction_dir = tmp_path / 'extract'
    escaped_path = tmp_path / 'escaped.txt'

    with tarfile.open(archive_path, mode) as archive:
        payload = b'escaped'
        member = tarfile.TarInfo(name='../escaped.txt')
        member.size = len(payload)
        archive.addfile(member, io.BytesIO(payload))

    with pytest.raises(tarfile.OutsideDestinationError):
        DownloadUtil.unzip(str(archive_path), str(extraction_dir))

    assert not escaped_path.exists()


def test_unzip_rejects_tar_link_outside_destination(tmp_path):
    archive_path = tmp_path / 'malicious-link.tar'
    extraction_dir = tmp_path / 'extract'

    with tarfile.open(archive_path, 'w') as archive:
        member = tarfile.TarInfo(name='outside-link')
        member.type = tarfile.SYMTYPE
        member.linkname = '../escaped.txt'
        archive.addfile(member)

    with pytest.raises(tarfile.LinkOutsideDestinationError):
        DownloadUtil.unzip(str(archive_path), str(extraction_dir))

    assert not (extraction_dir / 'outside-link').exists()


def test_download_from_google_drive_and_import_errors(tmp_path):
    fake_gdown = SimpleNamespace(download=lambda url, output, quiet: output)
    with patch.object(DownloadUtil, '_import_gdown', return_value=fake_gdown):
        output = DownloadUtil.download_from_google_drive(dest_dir=str(tmp_path), dest_file='model.bin')
    assert output.endswith('/model.bin')

    with patch.dict('sys.modules', {'gdown': None}):
        with pytest.raises(ModuleNotFoundError):
            DownloadUtil._import_gdown()


def test_download_model_shortcuts_and_invalid_names(tmp_path):
    with patch.object(DownloadUtil, 'download_from_google_drive', return_value=str(tmp_path / 'GoogleNews-vectors-negative300.bin.gz')) as mocked_drive, \
         patch.object(DownloadUtil, 'unzip') as mocked_unzip:
        DownloadUtil.download_word2vec(dest_dir=str(tmp_path))
        mocked_drive.assert_called_once()
        mocked_unzip.assert_called_once()

    with patch.object(DownloadUtil, 'download', return_value=str(tmp_path / 'glove.zip')) as mocked_download, \
         patch.object(DownloadUtil, 'unzip') as mocked_unzip:
        DownloadUtil.download_glove('glove.6B', str(tmp_path))
        mocked_download.assert_called_once_with(
            'https://nlp.stanford.edu/data/glove.6B.zip',
            dest_dir=str(tmp_path),
        )
        mocked_unzip.assert_called_once()

    with patch.object(DownloadUtil, 'download', return_value=str(tmp_path / 'fasttext.zip')) as mocked_download, \
         patch.object(DownloadUtil, 'unzip') as mocked_unzip:
        DownloadUtil.download_fasttext('wiki-news-300d-1M', str(tmp_path))
        mocked_download.assert_called_once()
        mocked_unzip.assert_called_once()

    with patch.object(DownloadUtil, 'download', return_value=str(tmp_path / 'back_translation.zip')) as mocked_download, \
         patch.object(DownloadUtil, 'unzip') as mocked_unzip:
        DownloadUtil.download_back_translation(str(tmp_path))
        mocked_download.assert_called_once()
        mocked_unzip.assert_called_once()

    with pytest.raises(ValueError):
        DownloadUtil.download_glove('unknown', str(tmp_path))

    with pytest.raises(ValueError):
        DownloadUtil.download_fasttext('unknown', str(tmp_path))
