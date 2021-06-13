
def new_id():
    import os
    from binascii import b2a_hex
    buf = os.urandom(16)
    return u'-'.join(b2a_hex(x).decode('ascii') for x in (
        buf[:4], buf[4:]
    ))

def new_id_bytes():
    return new_id().encode('ascii')

def ensure_dir(file_path):
    import os
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

def create_connection_file():
    import os
    import uuid
    from jupyter_core.paths import jupyter_runtime_dir
    from ipython_genutils.py3compat import unicode_type
    from jupyter_client.connect import write_connection_file
    kernel_id = unicode_type(uuid.uuid4())
    connection_file=os.path.join(jupyter_runtime_dir(), "kernel-%s.json" % kernel_id)
    ensure_dir(connection_file)
    write_connection_file(connection_file, key=new_id_bytes())
    return connection_file
