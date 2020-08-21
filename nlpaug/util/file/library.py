import os

import nlpaug


class LibraryUtil:
    """
    Helper function for retreiving library file

    >>> from nlpaug.util.file.library import LibraryUtil
    """

    @staticmethod
    def get_res_dir():
        """
        >>> LibraryUtil.get_res_dir()

        """
        lib_dir = os.path.dirname(nlpaug.__file__)
        return os.path.join(lib_dir, 'res')
