from pro_notebook_integration.handlers import setup_handlers

# Jupyter Extension points
def _jupyter_server_extension_paths():
    return [{
        'module': 'pro_notebook_integration',
    }]

def load_jupyter_server_extension(nbapp):
    setup_handlers(nbapp.web_app)
