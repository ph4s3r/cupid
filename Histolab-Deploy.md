
# Process of setting up a development environment with histolab for windows

based on https://histolab.readthedocs.io/en/latest/installation.html

- conda create --name histolab python=3.9 #reason of using python 3.9 because OpenCV had some issues on python 3.11
- isntall msys2 from https://www.msys2.org/ direct dl link: https://github.com/msys2/msys2-installer/releases/download/2023-10-26/msys2-x86_64-20231026.exe
- hit "pacman -S mingw-w64-x86_64-pixman" in the msys2 shell to install Pixman 0.40
- to install openslide on windows download the binary and extract to c:/openslide or somewhere or hit pacman -S mingw-w64-x86_64-openslide to install openslide (no conda packages available for win64! - the python packages are all for osx and linux etc, must be first checked with  the conda-forge channel browser)
- the point is that openslide is shipped with an older version of pixman so need to replace:
- cp C:\msys64\mingw64\bin\libpixman-1-0.dll C:\openslide-win64-20231011\bin\libpixman-1-0.dll # or
- cp C:\msys64\mingw64\bin\libpixman-1-0.dll C:\OpenSlide\bin\libpixman-1-0.dll
- also need to copy the openslide dll and rename the copy from ***1*** to ***0***
- pip install histolab or conda install -c conda-forge histolab