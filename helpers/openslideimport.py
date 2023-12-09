# The path can also be read from a config file, etc.
OPENSLIDE_PATH = "C:\\openslide-win64-20231011\\bin"

import os
if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    import openslide