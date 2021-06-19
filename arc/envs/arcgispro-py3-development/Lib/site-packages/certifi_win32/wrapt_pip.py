from __future__ import absolute_import
import sys
import wrapt


# Directly overriding download.HTTPAdapter working in debugging, not from console
# Ignore it, just fall through to the certifi patch for now
if False and sys.version_info >= (3, 4, 0):
    @wrapt.when_imported('pip._internal.download')
    def apply_patches(download):

        class SslContextHttpAdapter(download.HTTPAdapter):
            """Transport adapter that allows us to use system-provided SSL
            certificates."""

            def init_poolmanager(self, *args, **kwargs):
                import ssl
                ssl_context = ssl.create_default_context()
                ssl_context.load_default_certs()
                kwargs['ssl_context'] = ssl_context
                return super().init_poolmanager(*args, **kwargs)

        def wrapper(wrapped, instance, args, kwargs):
            retries = kwargs.get("retries", 0)
            wrapped(*args, **kwargs)
            secure_adapter = SslContextHttpAdapter(max_retries=retries)
            instance.mount("https://", secure_adapter)

        wrapt.wrap_function_wrapper(download.PipSession, '__init__', wrapper)

else:
    # Just patch the bundled certifi in pip >= 9.0.2
    @wrapt.when_imported('pip._vendor.requests.certs')
    def apply_patches(certs):
        import certifi
        certs.where = certifi.where
