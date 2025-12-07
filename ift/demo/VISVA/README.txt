//***************************************************//

BIA - Brain Image Analyzer
LIV, IC, Unicamp


Last Modified in Aug 12, 2010
//***************************************************//

Installation:

(1) Get the modules "ift" and "libopf" from the CVS repository.

(2) Set the environment variable in .bashrc file:
    Example:
      export IFT_DIR=/home/pavm/ift
      export OPF_DIR=/home/pavm/libopf
      export BIA_DIR=/home/pavm/bia
    Restart your bash terminal.

(3) Instale a biblioteca zlib1g-dev

(4) Go to the "bia" directory and execute:
    make
    ./bia


----------------------

INSTRUCTIONS FOR WINDOWS:

- Download mingw (exe installer) and install it 

  http://downloads.sourceforge.net/project/mingw/Automated%20MinGW%20Installer/MinGW%205.1.6/MinGW-5.1.6.exe


- Download msys (exe installer) and install it

  http://downloads.sourceforge.net/mingw/MSYS-1.0.11.exe


- Download wx-widgets wxMSW source-code (exe installer)

  http://prdownloads.sourceforge.net/wxwindows/wxMSW-2.8.11-Setup.exe

  ./configure --disable-shared --enable-unicode
  make
  make install

- Download pthreads for windows (ftp://sourceware.org/pub/pthreads-win32/pthreads-w32-2-8-0-release.exe)
  - compile it statically (make GC-static)
  - copy libpthreadGC2.a to libpthread.a to system "lib" folder (/mingw/lib)
  - copy header files to system "include" folder (/mingw/include)

- Open msys and set environment variables as in step (2)

- Do step (3)

- To generate an installer, use InstallJammer (www.installjammer.com/download)




