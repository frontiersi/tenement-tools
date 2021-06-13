from notebook.services.contents.filemanager import FileContentsManager
from notebook.utils import url_path_join, url_escape, to_os_path
from contextlib import contextmanager
from .ProCheckpoints import ProFileCheckpoints
from tornado import web
import nbformat
import base64
import os, io
import mimetypes
from datetime import datetime
from notebook import _tz as tz
from traitlets import default

class ProFileContentsManager(FileContentsManager):
    def pro_to_os_path(self, api_path):
        return to_os_path(api_path, root=self.notebook_dir)

    def _base_model(self, path):
        """Build the common base of a contents model"""
        os_path = self._get_os_path(path)
        info = os.lstat(os_path)

        try:
            # size of file
            size = info.st_size
        except (ValueError, OSError):
            self.log.warning('Unable to get size.')
            size = None

        try:
            last_modified = tz.utcfromtimestamp(info.st_mtime)
        except (ValueError, OSError):
            # Files can rarely have an invalid timestamp
            # https://github.com/jupyter/notebook/issues/2539
            # https://github.com/jupyter/notebook/issues/2757
            # Use the Unix epoch as a fallback so we don't crash.
            self.log.warning('Invalid mtime %s for %s', info.st_mtime, os_path)
            last_modified = datetime(1970, 1, 1, 0, 0, tzinfo=tz.UTC)

        try:
            created = tz.utcfromtimestamp(info.st_ctime)
        except (ValueError, OSError):  # See above
            self.log.warning('Invalid ctime %s for %s', info.st_ctime, os_path)
            created = datetime(1970, 1, 1, 0, 0, tzinfo=tz.UTC)

        # Create the base model.
        model = {}
        model['name'] = path.rsplit('/', 1)[-1]
        model['path'] = path
        model['last_modified'] = last_modified
        model['created'] = created
        model['content'] = None
        model['format'] = None
        model['mimetype'] = None
        model['size'] = size

        try:
            model['writable'] = os.access(os_path, os.W_OK)
        except OSError:
            self.log.error("Failed to check write permissions on %s", os_path)
            model['writable'] = False
        return model

    def _file_model(self, path, content=True, format=None):
        model = self._base_model(path)
        model['type'] = 'file'

        os_path = self._get_os_path(path)
        model['mimetype'] = mimetypes.guess_type(os_path)[0]

        if content:
            content, format = self._read_file(os_path, format)
            if model['mimetype'] is None:
                default_mime = {
                    'text': 'text/plain',
                    'base64': 'application/octet-stream'
                }[format]
                model['mimetype'] = default_mime

            model.update(
                content=content,
                format=format,
            )

        return model

    def _get_os_path(self, path):
        if os.path.isabs(path):
            return path
        root = os.path.abspath(self.root_dir)
        os_path = to_os_path(path, root)
        try:
            with self.open(path, 'r', encoding='utf-8') as f:
                pass
        except:
            print("invalid path: {}".format(path))
            return os_path
        return path

    def exists(self, path):
        """Returns True if the path exists, else returns False.

        API-style wrapper for os.path.exists

        Parameters
        ----------
        path : string
            The API path to the file (with '/' as separator)

        Returns
        -------
        exists : bool
            Whether the target exists.
        """
        path = path.strip('/')
        os_path = self._get_os_path(path=path)
        try:
            os.lstat(os_path)
        except OSError:
            return False
        return True

    def get(self, path, content=True, type=None, format=None):
        path = path.strip('/')

        if not self.exists(path):
            raise web.HTTPError(404, u'No such file or directory: %s' % path)

        os_path = self._get_os_path(path)
        if os.path.isdir(os_path):
            if type not in (None, 'directory'):
                raise web.HTTPError(400,
                                u'%s is a directory, not a %s' % (path, type), reason='bad type')
            model = self._dir_model(path, content=content)
        elif type == 'notebook' or (type is None and path.endswith('.ipynb')):
            model = self._notebook_model(path, content=content)
            if 'content' in model.keys():
                if model['content']:
                    if 'metadata' in model['content'].keys():
                        if 'kernelspec' in model['content']['metadata'].keys():
                            pass
                        else:
                            model['content']['metadata'] = {"kernelspec":{'display_name': 'ArcGISPro', 'language': 'python', 'name': 'python3'}}
                    else:
                        model['content']['metadata'] = {"kernelspec":{'display_name': 'ArcGISPro', 'language': 'python', 'name': 'python3'}}
        else:
            if type == 'directory':
                raise web.HTTPError(400,
                                u'%s is not a directory' % path, reason='bad type')
            model = self._file_model(path, content=content, format=format)
        return model

class ProLargeFileManager(ProFileContentsManager):

    def save(self, model, path=''):
        """Save the file model and return the model with no content."""
        chunk = model.get('chunk', None)
        if chunk is not None:
            path = path.strip('/')

            if 'type' not in model:
                raise web.HTTPError(400, u'No file type provided')
            if model['type'] != 'file':
                raise web.HTTPError(400, u'File type "{}" is not supported for large file transfer'.format(model['type']))
            if 'content' not in model and model['type'] != 'directory':
                raise web.HTTPError(400, u'No file content provided')

            os_path = self._get_os_path(path)

            try:
                if chunk == 1:
                    self.log.debug("Saving %s", os_path)
                    self.run_pre_save_hook(model=model, path=path)
                    super(ProLargeFileManager, self)._save_file(os_path, model['content'], model.get('format'))
                else:
                    self._save_large_file(os_path, model['content'], model.get('format'))
            except web.HTTPError:
                raise
            except Exception as e:
                self.log.error(u'Error while saving file: %s %s', path, e, exc_info=True)
                raise web.HTTPError(500, u'Unexpected error while saving file: %s %s' % (path, e))

            model = self.get(path, content=False)

            # Last chunk
            if chunk == -1:
                self.run_post_save_hook(model=model, os_path=os_path)
            return model
        else:
            return super(ProLargeFileManager, self).save(model, path)

    def create_checkpoint(self, path):
        """Create a checkpoint."""
        # return without creating checkpoint
        if path.startswith('/') or path.startswith('\\'):
            return self.checkpoints.create_checkpoint(self, path[1:])
        return self.checkpoints.create_checkpoint(self, path)

    @default('checkpoints_class')
    def _checkpoints_class_default(self):
        return ProFileCheckpoints

    def _save_large_file(self, os_path, content, format):
        """Save content of a generic file."""
        if format not in {'text', 'base64'}:
            raise web.HTTPError(
                400,
                "Must specify format of file contents as 'text' or 'base64'",
            )
        try:
            if format == 'text':
                bcontent = content.encode('utf8')
            else:
                b64_bytes = content.encode('ascii')
                bcontent = base64.b64decode(b64_bytes)
        except Exception as e:
            raise web.HTTPError(
                400, u'Encoding error saving %s: %s' % (os_path, e)
            )

        with self.perm_to_403(os_path):
            if os.path.islink(os_path):
                os_path = os.path.join(os.path.dirname(os_path), os.readlink(os_path))
            with io.open(os_path, 'ab') as f:
                f.write(bcontent)
