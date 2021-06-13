import os
import json
import logging
import subprocess as sp
from collections import namedtuple
from tornado import web

from notebook.utils import url_path_join as ujoin
from notebook.base.handlers import IPythonHandler, path_regex, FilesRedirectHandler

logger = logging.getLogger('ProProxy')

import gettext

# Set up message catalog access
base_dir = os.path.realpath(os.path.join(__file__, '..', '..'))
trans = gettext.translation('notebook', localedir=os.path.join(base_dir, 'notebook/i18n'), fallback=True)
_ = trans.gettext

def get_frontend_exporters():
    from nbconvert.exporters.base import get_export_names, get_exporter

    # name=exporter_name, display=export_from_notebook+extension
    ExporterInfo = namedtuple('ExporterInfo', ['name', 'display'])

    default_exporters = [
        ExporterInfo(name='html', display='HTML (.html)'),
        ExporterInfo(name='latex', display='LaTeX (.tex)'),
        ExporterInfo(name='markdown', display='Markdown (.md)'),
        ExporterInfo(name='notebook', display='Notebook (.ipynb)'),
        ExporterInfo(name='pdf', display='PDF via LaTeX (.pdf)'),
        ExporterInfo(name='rst', display='reST (.rst)'),
        ExporterInfo(name='script', display='Script (.txt)'),
        ExporterInfo(name='slides', display='Reveal.js slides (.slides.html)')
    ]

    frontend_exporters = []
    for name in get_export_names():
        exporter_class = get_exporter(name)
        exporter_instance = exporter_class()
        ux_name = getattr(exporter_instance, 'export_from_notebook', None)
        super_uxname = getattr(super(exporter_class, exporter_instance),
                               'export_from_notebook', None)

        # Ensure export_from_notebook is explicitly defined & not inherited
        if ux_name is not None and ux_name != super_uxname:
            display = _('{} ({})'.format(ux_name,
                                         exporter_instance.file_extension))
            frontend_exporters.append(ExporterInfo(name, display))

    # Ensure default_exporters are in frontend_exporters if not already
    # This protects against nbconvert versions lower than 5.5
    names = set(exporter.name.lower() for exporter in frontend_exporters)
    for exporter in default_exporters:
        if exporter.name not in names:
            frontend_exporters.append(exporter)

    # Protect against nbconvert 5.5.0
    python_exporter = ExporterInfo(name='python', display='python (.py)')
    if python_exporter in frontend_exporters:
        frontend_exporters.remove(python_exporter)

    # Protect against nbconvert 5.4.x
    template_exporter = ExporterInfo(name='custom', display='custom (.txt)')
    if template_exporter in frontend_exporters:
        frontend_exporters.remove(template_exporter)
    return sorted(frontend_exporters)


class ProProxyHandler(IPythonHandler):
    pro_port = 8005
    proc = None

    @web.authenticated
    def post(self):
        logger.info('%s request to %s', self.request.method, self.request.uri)
        response = {
            'pid':"0x001212",
            'url':'{}proxy/{}/'.format(self.base_url, self.pro_port),
        }
        self.finish(json.dumps(response))

    @web.authenticated
    def get(self):
        if not self.proc:
            self.set_status(500)
            self.write('rsession not yet started')
            self.finish()
        self.finish(self.proc.poll())

    def delete(self):
        logger.info('%s request to %s', self.request.method, self.request.uri)
        if self.pro:
            self.proc.kill()
            self.finish(self.proc.poll())

class ProNotebookHandler(IPythonHandler):

    @web.authenticated
    def get(self, path):
        path = path.strip('/')
        cm = self.contents_manager
        print(path)
        # will raise 404 on not found
        try:
            model = cm.get(path, content=False)
            print(model)
        except web.HTTPError as e:
            if e.status_code == 404 and 'files' in path.split('/'):
                # 404, but '/files/' in URL, let FilesRedirect take care of it
                return FilesRedirectHandler.redirect_to_files(self, path)
            else:
                raise
        if model['type'] != 'notebook':
            # not a notebook, redirect to files
            return FilesRedirectHandler.redirect_to_files(self, path)
        name = path.rsplit('/', 1)[-1]
        self.write(self.render_template('notebook.html',
            notebook_path=path,
            notebook_name=name,
            kill_kernel=False,
            mathjax_url=self.mathjax_url,
            mathjax_config=self.mathjax_config,
            get_frontend_exporters=get_frontend_exporters
            )
        )

def setup_handlers(web_app):
    host_pattern = '.*$'
    route_pattern = ujoin(web_app.settings['base_url'], '/proproxy/?')
    pro_notebook_route = ujoin(web_app.settings['base_url'], '/proproxy/?')
    web_app.add_handlers(host_pattern, [
        (route_pattern, ProProxyHandler),
        (r"/proproxy/notebooks%s" % path_regex, ProNotebookHandler)
    ])
    logger.info('Added handler for route %s', route_pattern)

#
