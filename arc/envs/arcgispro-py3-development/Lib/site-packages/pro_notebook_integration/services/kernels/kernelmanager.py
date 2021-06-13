from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
import os
from subprocess import Popen, PIPE
from tornado import gen, web
from tornado.concurrent import Future
from tornado.ioloop import IOLoop, PeriodicCallback

from jupyter_client.session import Session
from traitlets import (Any, Bool, Dict, List, Unicode, TraitError, Integer,
       Float, Instance, default, validate
)

from notebook.utils import maybe_future, to_os_path, exists
from notebook._tz import utcnow, isoformat
from ipython_genutils.py3compat import getcwd
from notebook.services.kernels.kernelmanager import MappingKernelManager
from jupyter_client.multikernelmanager import MultiKernelManager
from jupyter_client.localinterfaces import is_local_ip, local_ips
from ipython_genutils.encoding import getdefaultencoding
from ipython_genutils.py3compat import cast_bytes_py2, PY3
from traitlets.log import get_logger
import sys
from functools import wraps
import ctypes

PROCNAME = "ArcGISPro.exe"

def get_proc_by_id(process_id):
    import psutil
    try:
        proc = psutil.Process(process_id)
        if proc.name() == PROCNAME:
            return proc
    except:
        pass
    return None


def create_interrupt_event():
    """Create an interrupt event handle.

    The parent process should call this to create the
    interrupt event that is passed to the child process. It should store
    this handle and use it with ``send_interrupt`` to interrupt the child
    process.
    """
    # Create a security attributes struct that permits inheritance of the
    # handle by new processes.
    # FIXME: We can clean up this mess by requiring pywin32 for IPython.
    class SECURITY_ATTRIBUTES(ctypes.Structure):
        _fields_ = [ ("nLength", ctypes.c_int),
                     ("lpSecurityDescriptor", ctypes.c_void_p),
                     ("bInheritHandle", ctypes.c_int) ]
    sa = SECURITY_ATTRIBUTES()
    sa_p = ctypes.pointer(sa)
    sa.nLength = ctypes.sizeof(SECURITY_ATTRIBUTES)
    sa.lpSecurityDescriptor = 0
    sa.bInheritHandle = 1

    return ctypes.windll.kernel32.CreateEventA(
        sa_p,  # lpEventAttributes
        False, # bManualReset
        False, # bInitialState
        '')    # lpName

def send_interrupt(interrupt_handle):
    """ Sends an interrupt event using the specified handle.
    """
    ctypes.windll.kernel32.SetEvent(interrupt_handle)

def attach_kernel(cmd, cid=None, rdir=None, stdin=None, stdout=None, stderr=None, env=None,
                  independent=False, cwd=None, process_id=None, **kw):

    redirect_in = False#True
    _stdin = PIPE if stdin is None else stdin
    redirect_out = True #sys.executable.endswith('pythonw.exe')
    if redirect_out:
        blackhole = open(os.devnull, 'w')
        _stdout = blackhole if stdout is None else stdout
        _stderr = blackhole if stderr is None else stderr
    else:
        _stdout, _stderr = stdout, stderr

    env = env if (env is not None) else os.environ.copy()

    encoding = getdefaultencoding(prefer_stream=False)
    kwargs = kw.copy()
    main_args = dict(
        stdin=_stdin,
        stdout=_stdout,
        stderr=_stderr,
        cwd=cwd,
        env=env,
    )
    kwargs.update(main_args)

    # Spawn a kernel.
    if sys.platform == 'win32':
        # Popen on Python 2 on Windows cannot handle unicode args or cwd
        cmd = [ cast_bytes_py2(c, encoding) for c in cmd ]
        if cwd:
            cwd = cast_bytes_py2(cwd, sys.getfilesystemencoding() or 'ascii')
            kwargs['cwd'] = cwd

        # Create a Win32 event for interrupting the kernel
        # and store it in an environment variable.
        interrupt_event = create_interrupt_event()
        env["JPY_INTERRUPT_EVENT"] = str(interrupt_event)
        # deprecated old env name:
        env["IPY_INTERRUPT_EVENT"] = env["JPY_INTERRUPT_EVENT"]

        try:
            from _winapi import DuplicateHandle, GetCurrentProcess, \
                DUPLICATE_SAME_ACCESS, CREATE_NEW_PROCESS_GROUP
        except:
            from _subprocess import DuplicateHandle, GetCurrentProcess, \
                DUPLICATE_SAME_ACCESS, CREATE_NEW_PROCESS_GROUP

        # create a handle on the parent to be inherited
        if independent:
            kwargs['creationflags'] = CREATE_NEW_PROCESS_GROUP
        else:
            pid = GetCurrentProcess()
            handle = DuplicateHandle(pid, pid, pid, 0,
                                     True, # Inheritable by new processes.
                                     DUPLICATE_SAME_ACCESS)
            env['JPY_PARENT_PID'] = str(int(handle))

        # Prevent creating new console window on pythonw
        if redirect_out:
            kwargs['creationflags'] = kwargs.setdefault('creationflags', 0) | 0x08000000 # CREATE_NO_WINDOW

        # Avoid closing the above parent and interrupt handles.
        # close_fds is True by default on Python >=3.7
        # or when no stream is captured on Python <3.7
        # (we always capture stdin, so this is already False by default on <3.7)
        kwargs['close_fds'] = False
    else:
        # Create a new session.
        # This makes it easier to interrupt the kernel,
        # because we want to interrupt the whole process group.
        # We don't use setpgrp, which is known to cause problems for kernels starting
        # certain interactive subprocesses, such as bash -i.
        if PY3:
            kwargs['start_new_session'] = True
        else:
            kwargs['preexec_fn'] = lambda: os.setsid()
        if not independent:
            env['JPY_PARENT_PID'] = str(os.getpid())

    try:
        import psutil
        def poller(proc_id):
            try:
                psutil.Process(proc_id)
            except:
                try:
                    info_file = "nbserver-%s.json" % os.getpid()
                    open_file = "nbserver-%s-open.html" % os.getpid()
                    os.unlink(cid)
                    os.unlink(os.path.join(rdir, info_file))
                    os.unlink(os.path.join(rdir, open_file))
                except:
                    pass
                sys.exit()
            return None
        def poller_wrapper():
            return poller(process_id)

        proc = get_proc_by_id(process_id)
        is_valid_proc = True
        if proc is None:
            is_valid_proc = False
            proc = Popen(cmd, **kwargs)
        proc.stdin = None
        proc.stdout = None
        proc.stderr = None
        if is_valid_proc:
            proc.poll = poller_wrapper

    except Exception as exc:
        msg = (
            "Failed to run command:\n{}\n"
            "    PATH={!r}\n"
            "    with kwargs:\n{!r}\n"
        )
        # exclude environment variables,
        # which may contain access tokens and the like.
        without_env = {key:value for key, value in kwargs.items() if key != 'env'}
        msg = msg.format(cmd, env.get('PATH', os.defpath), without_env)
        get_logger().error(msg)
        raise

    if sys.platform == 'win32':
        # Attach the interrupt event to the Popen objet so it can be used later.
        proc.win32_interrupt_event = interrupt_event

    # Clean up pipes created to work around Popen bug.
    if redirect_in:
        if stdin is None:
            proc.stdin.close()

    return proc

class ProMappingKernelManager(MappingKernelManager):
    """A KernelManager that handles notebook mapping and HTTP error handling"""

    @default('kernel_manager_class')
    def _default_kernel_manager_class(self):
        return "jupyter_client.ioloop.IOLoopKernelManager"

    kernel_argv = List(Unicode())

    root_dir = Unicode(config=True)

    _kernel_connections = Dict()

    _culler_callback = None

    _initialized_culler = False

    _load_esri_config_file = True

    _pro_kernel_name = "ArcGISPro"

    @default('root_dir')
    def _default_root_dir(self):
        try:
            return self.parent.notebook_dir
        except AttributeError:
            return getcwd()

    @validate('root_dir')
    def _update_root_dir(self, proposal):
        """Do a bit of validation of the root dir."""
        value = proposal['value']
        if not os.path.isabs(value):
            # If we receive a non-absolute path, make it absolute.
            value = os.path.abspath(value)
        if not exists(value) or not os.path.isdir(value):
            raise TraitError("kernel root dir %r is not a directory" % value)
        return value

    cull_idle_timeout = Integer(0, config=True,
        help="""Timeout (in seconds) after which a kernel is considered idle and ready to be culled.
        Values of 0 or lower disable culling. Very short timeouts may result in kernels being culled
        for users with poor network connections."""
    )

    cull_interval_default = 300 # 5 minutes
    cull_interval = Integer(cull_interval_default, config=True,
        help="""The interval (in seconds) on which to check for idle kernels exceeding the cull timeout value."""
    )

    cull_connected = Bool(False, config=True,
        help="""Whether to consider culling kernels which have one or more connections.
        Only effective if cull_idle_timeout > 0."""
    )

    cull_busy = Bool(False, config=True,
        help="""Whether to consider culling kernels which are busy.
        Only effective if cull_idle_timeout > 0."""
    )

    buffer_offline_messages = Bool(True, config=True,
        help="""Whether messages from kernels whose frontends have disconnected should be buffered in-memory.

        When True (default), messages are buffered and replayed on reconnect,
        avoiding lost messages due to interrupted connectivity.

        Disable if long-running kernels will produce too much output while
        no frontends are connected.
        """
    )

    kernel_info_timeout = Float(60, config=True,
        help="""Timeout for giving up on a kernel (in seconds).

        On starting and restarting kernels, we check whether the
        kernel is running and responsive by sending kernel_info_requests.
        This sets the timeout in seconds for how long the kernel can take
        before being presumed dead.
        This affects the MappingKernelManager (which handles kernel restarts)
        and the ZMQChannelsHandler (which handles the startup).
        """
    )

    _kernel_buffers = Any()
    @default('_kernel_buffers')
    def _default_kernel_buffers(self):
        return defaultdict(lambda: {'buffer': [], 'session_key': '', 'channels': {}})

    last_kernel_activity = Instance(datetime,
        help="The last activity on any kernel, including shutting down a kernel")

    def __init__(self, **kwargs):
        super(ProMappingKernelManager, self).__init__(**kwargs)
        self.last_kernel_activity = utcnow()

    allowed_message_types = List(trait=Unicode(), config=True,
        help="""White list of allowed kernel message types.
        When the list is empty, all message types are allowed.
        """
    )

    #-------------------------------------------------------------------------
    # Methods for managing kernels and sessions
    #-------------------------------------------------------------------------

    def _handle_kernel_died(self, kernel_id):
        """notice that a kernel died"""
        self.log.warning("Kernel %s died, removing from map.", kernel_id)
        self.remove_kernel(kernel_id)

    def cwd_for_path(self, path):
        """Turn API path into absolute OS path."""
        os_path = to_os_path(path, self.root_dir)
        # in the case of notebooks and kernels not being on the same filesystem,
        # walk up to root_dir if the paths don't exist
        while not os.path.isdir(os_path) and os_path != self.root_dir:
            os_path = os.path.dirname(os_path)
        return os_path

    def get_proc_id(self, kernel_name=None, **kwargs):
        try:
            return self.parent.pid
        except:
            raise RuntimeError("Cannot find process without id")

    def get_pro_connection_file_path(self):
        try:
            return self.parent.cid
        except:
            raise RuntimeError("Cannot find connection file")

    def get_pro_kernel_id(self):
        try:
            connection_path = self.parent.cid
            import ntpath
            head, tail = ntpath.split(connection_path)
            if tail:
                return tail.split('kernel-')[1].split('.')[0]
            else:
                raise RuntimeError("Cannot find kernel id")
        except:
            raise RuntimeError("Cannot get kernel_id")


    def pro_start_kernel(self, kernel_name=None, **kwargs):
        kernel_id = ""

        if self._load_esri_config_file:
            kernel_id = self.get_pro_kernel_id()
        else:
            kernel_id = kwargs.pop('kernel_id', self.new_kernel_id(**kwargs))

        #if kernel_id in self:
        #    raise DuplicateKernelError('Kernel already exists: %s' % kernel_id)
        if kernel_name is None:
            kernel_name = self._pro_kernel_name

        constructor_kwargs = {'cache_ports': False}
        if self.kernel_spec_manager:
            constructor_kwargs['kernel_spec_manager'] = self.kernel_spec_manager

        km = self.kernel_manager_factory(connection_file=os.path.join(
                    self.connection_dir, "kernel-%s.json" % kernel_id),
                    parent=self, log=self.log, kernel_name=kernel_name,
                    **constructor_kwargs
        )
        if km.transport == 'tcp' and not is_local_ip(km.ip):
            raise RuntimeError("Can only launch a kernel on a local interface. "
                           "This one is not: %s."
                           "Make sure that the '*_address' attributes are "
                           "configured properly. "
                           "Currently valid addresses are: %s" % (km.ip, local_ips()))

        if self._load_esri_config_file:
            km.load_connection_file(self.get_pro_connection_file_path())
        else:
            km.write_connection_file()

        km._launch_args = kwargs.copy()
        extra_arguments = kwargs.pop('extra_arguments', [])
        kernel_cmd = km.format_kernel_cmd(extra_arguments=extra_arguments)
        env = kwargs.pop('env', os.environ).copy()
        if not km.kernel_cmd:
            env.update(km.kernel_spec.env or {})
        elif km.extra_env:
            env.update(km.extra_env)

        km.log.debug("Starting kernel: %s", kernel_cmd)
        if self._load_esri_config_file:
            km.kernel = attach_kernel(kernel_cmd, cid=self.parent.cid, rdir=self.parent.runtime_dir, env=env, process_id=self.get_proc_id(), **kwargs)
        else:
            km.kernel = km._launch_kernel(kernel_cmd, env=env, **kwargs)

        km.start_restarter()
        km._connect_control_socket()
        self._kernels[kernel_id] = km
        return kernel_id

    @gen.coroutine
    def start_kernel(self, kernel_id=None, path=None, **kwargs):
        if kernel_id is None:
            if path is not None:
                kwargs['cwd'] = self.cwd_for_path(path)
            kernel_id = yield maybe_future(
                self.pro_start_kernel(**kwargs)
            )
            self._kernel_connections[kernel_id] = 0
            self.start_watching_activity(kernel_id)
            self.log.info("Kernel started: %s" % kernel_id)
            self.log.debug("Kernel args: %r" % kwargs)
            # register callback for failed auto-restart
            self.add_restart_callback(kernel_id,
                lambda : self._handle_kernel_died(kernel_id),
                'dead',
            )
        else:
            self._check_kernel_id(kernel_id)
            self.log.info("Using existing kernel: %s" % kernel_id)

        # Initialize culling if not already
        if not self._initialized_culler:
            self.initialize_culler()

        # py2-compat
        raise gen.Return(kernel_id)

    def create_checkpoint(self, path):
        """Create a checkpoint."""
        return self.checkpoints.create_checkpoint(self, path)

    def start_buffering(self, kernel_id, session_key, channels):
        """Start buffering messages for a kernel

        Parameters
        ----------
        kernel_id : str
            The id of the kernel to stop buffering.
        session_key: str
            The session_key, if any, that should get the buffer.
            If the session_key matches the current buffered session_key,
            the buffer will be returned.
        channels: dict({'channel': ZMQStream})
            The zmq channels whose messages should be buffered.
        """

        if not self.buffer_offline_messages:
            for channel, stream in channels.items():
                stream.close()
            return

        self.log.info("Starting buffering for %s", session_key)
        self._check_kernel_id(kernel_id)
        # clear previous buffering state
        self.stop_buffering(kernel_id)
        buffer_info = self._kernel_buffers[kernel_id]
        # record the session key because only one session can buffer
        buffer_info['session_key'] = session_key
        # TODO: the buffer should likely be a memory bounded queue, we're starting with a list to keep it simple
        buffer_info['buffer'] = []
        buffer_info['channels'] = channels

        # forward any future messages to the internal buffer
        def buffer_msg(channel, msg_parts):
            self.log.debug("Buffering msg on %s:%s", kernel_id, channel)
            buffer_info['buffer'].append((channel, msg_parts))

        for channel, stream in channels.items():
            stream.on_recv(partial(buffer_msg, channel))

    def get_buffer(self, kernel_id, session_key):
        """Get the buffer for a given kernel

        Parameters
        ----------
        kernel_id : str
            The id of the kernel to stop buffering.
        session_key: str, optional
            The session_key, if any, that should get the buffer.
            If the session_key matches the current buffered session_key,
            the buffer will be returned.
        """
        self.log.debug("Getting buffer for %s", kernel_id)
        if kernel_id not in self._kernel_buffers:
            return

        buffer_info = self._kernel_buffers[kernel_id]
        if buffer_info['session_key'] == session_key:
            # remove buffer
            self._kernel_buffers.pop(kernel_id)
            # only return buffer_info if it's a match
            return buffer_info
        else:
            self.stop_buffering(kernel_id)

    def stop_buffering(self, kernel_id):
        """Stop buffering kernel messages

        Parameters
        ----------
        kernel_id : str
            The id of the kernel to stop buffering.
        """
        self.log.debug("Clearing buffer for %s", kernel_id)
        self._check_kernel_id(kernel_id)

        if kernel_id not in self._kernel_buffers:
            return
        buffer_info = self._kernel_buffers.pop(kernel_id)
        # close buffering streams
        for stream in buffer_info['channels'].values():
            if not stream.closed():
                stream.on_recv(None)
                stream.close()

        msg_buffer = buffer_info['buffer']
        if msg_buffer:
            self.log.info("Discarding %s buffered messages for %s",
                len(msg_buffer), buffer_info['session_key'])

    def shutdown_kernel(self, kernel_id, now=False):
        """Shutdown a kernel by kernel_id"""
        return

    @gen.coroutine
    def restart_kernel(self, kernel_id):
        """Restart a kernel by kernel_id"""
        #Should not be possible to do this
        return


    def notify_connect(self, kernel_id):
        """Notice a new connection to a kernel"""
        if kernel_id in self._kernel_connections:
            self._kernel_connections[kernel_id] += 1

    def notify_disconnect(self, kernel_id):
        """Notice a disconnection from a kernel"""
        if kernel_id in self._kernel_connections:
            self._kernel_connections[kernel_id] -= 1

    def kernel_model(self, kernel_id):
        """Return a JSON-safe dict representing a kernel

        For use in representing kernels in the JSON APIs.
        """
        self._check_kernel_id(kernel_id)
        kernel = self._kernels[kernel_id]

        model = {
            "id":kernel_id,
            "name": kernel.kernel_name,
            "last_activity": isoformat(kernel.last_activity),
            "execution_state": kernel.execution_state,
            "connections": self._kernel_connections[kernel_id],
        }
        return model

    def list_kernels(self):
        """Returns a list of kernel_id's of kernels running."""
        kernels = []
        kernel_ids = super(ProMappingKernelManager, self).list_kernel_ids()
        for kernel_id in kernel_ids:
            model = self.kernel_model(kernel_id)
            kernels.append(model)
        return kernels

    # override _check_kernel_id to raise 404 instead of KeyError
    def _check_kernel_id(self, kernel_id):
        """Check a that a kernel_id exists and raise 404 if not."""
        if kernel_id not in self:
            raise web.HTTPError(404, u'Kernel does not exist: %s' % kernel_id)

    # monitoring activity:

    def start_watching_activity(self, kernel_id):
        """Start watching IOPub messages on a kernel for activity.

        - update last_activity on every message
        - record execution_state from status messages
        """
        kernel = self._kernels[kernel_id]
        # add busy/activity markers:
        kernel.execution_state = 'starting'
        kernel.last_activity = utcnow()
        kernel._activity_stream = kernel.connect_iopub()
        session = Session(
            config=kernel.session.config,
            key=kernel.session.key,
        )

        def record_activity(msg_list):
            """Record an IOPub message arriving from a kernel"""
            self.last_kernel_activity = kernel.last_activity = utcnow()

            idents, fed_msg_list = session.feed_identities(msg_list)
            msg = session.deserialize(fed_msg_list)

            msg_type = msg['header']['msg_type']
            if msg_type == 'status':
                kernel.execution_state = msg['content']['execution_state']
                self.log.debug("activity on %s: %s (%s)", kernel_id, msg_type, kernel.execution_state)
            else:
                self.log.debug("activity on %s: %s", kernel_id, msg_type)

        kernel._activity_stream.on_recv(record_activity)

    def initialize_culler(self):
        """Start idle culler if 'cull_idle_timeout' is greater than zero.

        Regardless of that value, set flag that we've been here.
        """
        if not self._initialized_culler and self.cull_idle_timeout > 0:
            if self._culler_callback is None:
                loop = IOLoop.current()
                if self.cull_interval <= 0: #handle case where user set invalid value
                    self.log.warning("Invalid value for 'cull_interval' detected (%s) - using default value (%s).",
                        self.cull_interval, self.cull_interval_default)
                    self.cull_interval = self.cull_interval_default
                self._culler_callback = PeriodicCallback(
                    self.cull_kernels, 1000*self.cull_interval)
                self.log.info("Culling kernels with idle durations > %s seconds at %s second intervals ...",
                    self.cull_idle_timeout, self.cull_interval)
                if self.cull_busy:
                    self.log.info("Culling kernels even if busy")
                if self.cull_connected:
                    self.log.info("Culling kernels even with connected clients")
                self._culler_callback.start()

        self._initialized_culler = True

    def cull_kernels(self):
        self.log.debug("Polling every %s seconds for kernels idle > %s seconds...",
            self.cull_interval, self.cull_idle_timeout)
        """Create a separate list of kernels to avoid conflicting updates while iterating"""
        for kernel_id in list(self._kernels):
            try:
                self.cull_kernel_if_idle(kernel_id)
            except Exception as e:
                self.log.exception("The following exception was encountered while checking the idle duration of kernel %s: %s",
                    kernel_id, e)

    def cull_kernel_if_idle(self, kernel_id):
        kernel = self._kernels[kernel_id]
        self.log.debug("kernel_id=%s, kernel_name=%s, last_activity=%s", kernel_id, kernel.kernel_name, kernel.last_activity)
        if kernel.last_activity is not None:
            dt_now = utcnow()
            dt_idle = dt_now - kernel.last_activity
            # Compute idle properties
            is_idle_time = dt_idle > timedelta(seconds=self.cull_idle_timeout)
            is_idle_execute = self.cull_busy or (kernel.execution_state != 'busy')
            connections = self._kernel_connections.get(kernel_id, 0)
            is_idle_connected = self.cull_connected or not connections
            # Cull the kernel if all three criteria are met
            if (is_idle_time and is_idle_execute and is_idle_connected):
                idle_duration = int(dt_idle.total_seconds())
                self.log.warning("Culling '%s' kernel '%s' (%s) with %d connections due to %s seconds of inactivity.",
                                 kernel.execution_state, kernel.kernel_name, kernel_id, connections, idle_duration)
                self.shutdown_kernel(kernel_id)
