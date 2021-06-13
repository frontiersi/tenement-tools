from __future__ import absolute_import

import wrapt

certifi_where = None


def wrap_where(wrapped, instance, args, kwargs):
    import certifi_win32.wincerts

    return certifi_win32.wincerts.where()


@wrapt.when_imported('certifi')
def apply_patches(certifi):
    # Keep local copy of original certifi.where() function
    global certifi_where
    certifi_where = certifi.where
    import certifi_win32.wincerts
    certifi_win32.wincerts.CERTIFI_PEM = certifi.where()

    # Wrap the certify.where function
    wrapt.wrap_function_wrapper(certifi, 'where', wrap_where)

    from certifi_win32.wincerts import generate_pem, verify_combined_pem
    if not verify_combined_pem():
        generate_pem()
