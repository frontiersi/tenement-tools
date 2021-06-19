#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

"""
This module copies the local store certificates to the current cacert.pem
file and caches it localy.
"""

PEM_PATH = os.path.join(os.environ['LOCALAPPDATA'], '.certifi', 'cacert.pem')

CERTIFI_PEM = None


def where():
    return PEM_PATH


def get_pems_wincertstore(store_names=None):
    import wincertstore
    store_names = store_names or ("CA", "ROOT")
    for store_name in store_names:
        with wincertstore.CertSystemStore(store_name) as store:
            for cert in store.itercerts(usage=wincertstore.SERVER_AUTH):
                try:
                    pem = cert.get_pem()
                    pem_entry = '# Label: "{name}"\n{pem}'.format(
                        name=cert.get_name(),
                        pem=pem.decode('ascii') if isinstance(pem, bytes) else pem
                    )
                except UnicodeEncodeError:
                    pem_entry = ''

                yield pem_entry


def get_pems(store_names=None):
    try:
        import ssl
        ssl_context = ssl.create_default_context()
        ssl_context.load_default_certs()
        for der_cert in ssl_context.get_ca_certs(binary_form=True):
            yield(ssl.DER_cert_to_PEM_cert(der_cert))
    except AttributeError:
        return get_pems_wincertstore(store_names)


def certifi_pem():
    global CERTIFI_PEM
    if not CERTIFI_PEM:
        import certifi
        CERTIFI_PEM = os.path.join(os.path.split(certifi.__file__)[0], 'cacert.pem')
        if not os.path.exists(CERTIFI_PEM):
            raise ValueError("Cannot find certifi cacert.pem")
    return CERTIFI_PEM


def verify_combined_pem():
    existing_correct = False

    if os.path.exists(PEM_PATH):
        chunk_size = 32

        with open(certifi_pem()) as certifi_pem_handle:
            certifi_head = certifi_pem_handle.read(chunk_size)

            certifi_pem_handle.seek(0, os.SEEK_END)
            tail_pos = certifi_pem_handle.tell() - chunk_size
            certifi_pem_handle.seek(tail_pos)

            certifi_tail = certifi_pem_handle.read(chunk_size)

        if os.path.getsize(PEM_PATH) > (tail_pos + chunk_size):
            with open(PEM_PATH) as pem_path_handle:
                local_head = pem_path_handle.read(chunk_size)
                pem_path_handle.seek(tail_pos)
                local_tail = pem_path_handle.read(chunk_size)

            if local_head == certifi_head and local_tail == certifi_tail:
                existing_correct = True

    return existing_correct


def generate_pem():
    import ctypes
    from ctypes import wintypes

    # Create ctypes wrapper for Win32 functions we need, with correct argument/return types
    _CreateMutex = ctypes.windll.kernel32.CreateMutexA
    _CreateMutex.argtypes = [wintypes.LPCVOID, wintypes.BOOL, wintypes.LPCSTR]
    _CreateMutex.restype = wintypes.HANDLE

    _WaitForSingleObject = ctypes.windll.kernel32.WaitForSingleObject
    _WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    _WaitForSingleObject.restype = wintypes.DWORD

    _ReleaseMutex = ctypes.windll.kernel32.ReleaseMutex
    _ReleaseMutex.argtypes = [wintypes.HANDLE]
    _ReleaseMutex.restype = wintypes.BOOL

    _CloseHandle = ctypes.windll.kernel32.CloseHandle
    _CloseHandle.argtypes = [wintypes.HANDLE]
    _CloseHandle.restype = wintypes.BOOL

    INFINITE = 0xFFFFFFFF

    handle = _CreateMutex(None, False, b'global_certifi_win32')
    _WaitForSingleObject(handle, INFINITE)

    if not os.path.exists(PEM_PATH):
        os.makedirs(os.path.dirname(PEM_PATH))

    orig_pem = certifi_pem()
    import shutil
    shutil.copy(orig_pem, PEM_PATH)

    import codecs
    with codecs.open(PEM_PATH, 'a', 'utf-8') as f:
        for pem in get_pems():
            f.write(pem)

    _ReleaseMutex(handle)
    _CloseHandle(handle)


if __name__ == '__main__':
    print(PEM_PATH)
