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


def test_broken_gdown_install_raises_helpful_error(tmp_path):
    # Issue #310: a broken gdown/requests install raises at `import gdown`
    # time (e.g. `UnknownExtra: ... 'socks'`). Download helpers must surface
    # a helpful ImportError instead of the raw packaging error.
    class BrokenGdownError(Exception):
        pass

    with patch.dict('sys.modules', {'gdown': None}):
        # Simulate `import gdown` raising a non-ImportError packaging error.
        import builtins
        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == 'gdown' or (isinstance(name, str) and name.startswith('gdown.')):
                raise BrokenGdownError("requests 2.27.1 has no such extra feature 'socks'")
            return real_import(name, *args, **kwargs)

        with patch.object(builtins, '__import__', side_effect=fake_import):
            with pytest.raises(ImportError) as exc_info:
                DownloadUtil._import_gdown()
            assert 'gdown' in str(exc_info.value)
            assert isinstance(exc_info.value.__cause__, BrokenGdownError)

            with pytest.raises(ImportError):
                DownloadUtil.download_from_google_drive(dest_dir=str(tmp_path), dest_file='model.bin')


def test_importing_download_utils_does_not_import_gdown():
    # Issue #310: `import nlpaug...` must work even when gdown is broken,
    # so gdown must stay a lazy (function-level) import. Static check: no
    # top-level `import gdown` in the module source.
    import ast
    from pathlib import Path

    import nlpaug.util.file.download as download_module

    source = Path(download_module.__file__).read_text()
    tree = ast.parse(source)
    top_level_gdown_imports = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for name in getattr(node, 'names', [])
        if (getattr(name, 'name', '') or '').split('.')[0] == 'gdown'
        or (isinstance(node, ast.ImportFrom) and (node.module or '').split('.')[0] == 'gdown')
    ]
    assert top_level_gdown_imports == []


def test_package_imports_work_with_broken_gdown():
    # Behavioral proof for #310: importing the public augmenter/flow entry
    # points in a fresh interpreter with broken gdown must succeed.
    import subprocess
    import sys

    from pathlib import Path
    import nlpaug

    repo_root = Path(nlpaug.__file__).resolve().parent.parent
    code = '\n'.join([
        'import sys',
        'class B(Exception): pass',
        'class F:',
        '    def find_module(self, name, path=None):',
        '        return self if name == "gdown" or name.startswith("gdown.") else None',
        '    def load_module(self, name):',
        '        raise B("requests 2.27.1 has no such extra feature socks")',
        'sys.meta_path.insert(0, F())',
        'sys.modules.pop("gdown", None)',
        'import nlpaug.augmenter.word, nlpaug.flow',
        'from nlpaug.util.file.download import DownloadUtil',
        'print("imports ok")',
    ])
    proc = subprocess.run(
        [sys.executable, '-c', code],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert 'imports ok' in proc.stdout


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
