from threading import Thread
from notebook import notebookapp
from os.path import join
import asyncio
import sys, getopt
import os

from pro_notebook_integration.services.kernels.kernelmanager import ProMappingKernelManager
from pro_notebook_integration.services.contents.largefilemanager import ProLargeFileManager
from pro_notebook_integration.client.kernelspec import ProKernelSpecManager

def launch_notebook_app(proc_id, con_id, ip, port, token, password, is_required, theme, options):
    na = notebookapp.NotebookApp.instance()
    na.load_config_file()
    na.open_browser = False
    na.pid = int(proc_id)
    na.cid = con_id
    na.port = int(port)
    na.token = token
    na.password = password
    if options == "DEBUG":
        na.log_level = "DEBUG"
    na.ip = ip
    if theme == 'dark':
        na.static_custom_path = [join(sys.prefix, 'share', 'jupyter', 'kernels', 'ArcGISPro', 'dark')]
    else:
        na.static_custom_path = [join(sys.prefix, 'share', 'jupyter', 'kernels', 'ArcGISPro', 'default')]
    na.kernel_manager_class = ProMappingKernelManager
    na.contents_manager_class = ProLargeFileManager
    na.kernel_spec_manager_class = ProKernelSpecManager
    na.password_required = is_required
    na.launch_instance()

def pro_process_loop(external_pid, self_pid):
    import psutil
    import signal
    import time
    while psutil.pid_exists(external_pid):
        time.sleep(4)
    else:
        os.kill(self_pid, signal.CTRL_C_EVENT)

def launch_server(proc_id, con_id, ip, port, token, password, is_required, theme, options):
    asyncio.set_event_loop(asyncio.new_event_loop())
    launch_notebook_app(proc_id, con_id, ip, port, token, password, is_required, theme, options)

def clear_connections(connection_file):
    try:
        dir_path = os.path.dirname(connection_file)
        os.unlink(connection_file)
        os.unlink(os.path.join(dir_path, 'nbserver-{}.json'.format(os.getpid())))
        os.unlink(os.path.join(dir_path, 'nbserver-{}-open.html'.format(os.getpid())))
    except:
        pass

def start_thread(proc_id, con_id, port, token, password, is_required, theme, ip="localhost", options="RELEASE", props=None):
    import atexit
    atexit.register(clear_connections, connection_file=con_id)
    loop_thread = Thread(target=pro_process_loop, args=(proc_id, os.getpid()))
    loop_thread.start()
    server_thread = Thread(target=launch_server, args=(proc_id, con_id, ip,  port, token, password, is_required, theme, options))
    server_thread.daemon = True
    server_thread.run()

