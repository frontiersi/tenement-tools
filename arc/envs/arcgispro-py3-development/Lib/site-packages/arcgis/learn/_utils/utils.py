def extract_zipfile(filepath, filename, remove=False):
    """ Function to extract the contents of a zip file
        Args:
            filepath: absolute path to the file directory.
            filename: name of the zip file to be extracted.
            remove: default=False, removes the original zip file 
                    after extracting the contents if True
    """
    import os, zipfile
    with zipfile.ZipFile(os.path.join(filepath, filename),"r") as zip_ref:
        zip_ref.extractall(filepath)
    if remove:
        os.remove(os.path.join(filepath, filename))
