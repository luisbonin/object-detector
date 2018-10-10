import os
import tarfile

from urllib.request import URLopener


def download(url, filename=None, force_download=False):
    """
    Download one file and save as filename
    :param str url: Link of the file
    :param str filename: Name to save the file. If None the name will be the default specified by URL
    :param bool force_download: Download the file even if it is already in specified path
    :return: True if is successful; False otherwise
    :rtype: bool
    """
    if not force_download:
        try:
            path, filename = filename.rsplit('/', 1)
        except ValueError:
            path = None

        if filename in os.listdir(path):
            return True

    opener = URLopener()

    return opener.retrieve(url=url, filename=filename)


def extract(filename, specific_file=None):
    """
    Extract a full archive or a specific file of it
    :param str filename: Name of the archive
    :param str specific_file: Name o the specific component of the archive to extract. If None all the files will be
                              extracted
    :return: True if is successful; False otherwise
    :rtype: bool
    """
    tar_file = tarfile.open(name=filename)

    if specific_file:
        for file in tar_file.getmembers():
            file_name = os.path.basename(file.name)

            if specific_file in file_name:
                tar_file.extract(member=file)
    else:
        tar_file.extractall()

    return True
