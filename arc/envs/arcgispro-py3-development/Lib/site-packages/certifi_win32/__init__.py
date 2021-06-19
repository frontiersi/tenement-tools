# Can be called from command line to re-generate the combined pem eg.
# python -m python_certifi_win32
from .wincerts import generate_pem

if __name__ == '__main__':
    import certifi  # Ensure patch is in place
    generate_pem()
