# Installing VISVA

#### Dependencies:
* wxWidgets 3.0 (wxgtk)
* gcc >= 4.9 and <= 5.5
* g++ >= 4.9 and <= 5.5
* zlib
* g++
* libblas
* libatlas
* liblapack
* export LC_ALL="en_US.UTF-8" // just works with this locale

#### UBUNTU
```
Installing wx-widgets 2.8 on ubuntu
1) Download the wxGTK2.8: https://github.com/wxWidgets/wxWidgets/releases/download/v2.8.12/wxGTK-2.8.12.tar.gz
2) Extract it into a folder
3) Install GTK2.8
sudo apt-get install gtk2.0
sudo apt-get install build-essential libgtk2.0-dev
4) Compile wx
cd folder_where_wx_is
mkdir buildgtk
cd buildgtk
../configure --with-gtk
make
make install
cd ..

Possible errors:
- If an error like this happens when executing visva:
visva: error while loading shared libraries: libwx_gtk2_adv-2.8.so.0: cannot open shared object file: No such file or directory

- Install the 32-bit version of libtk2.0
sudo apt-get install libgtk2.0-0:i386


Uninstalling
sudo su
rm -f  /usr/bin/wx*
rm -r  /usr/include/wx*
rm -r  /usr/lib/wx
rm -f  /usr/lib/libwx*
rm -f  /usr/local/bin/wx*
rm -r  /usr/local/include/wx*
rm -r  /usr/local/lib/wx
rm -f  /usr/local/lib/libwx*
ldconfig

sudo apt-get install zlib1g-dev
sudo apt-get install g++
sudo apt-get install libblas-dev libatlas-dev liblapack-dev
export LC_ALL="en_US.UTF-8"
```

#### ARCH LINUX
```
$ yaourt -Su wxgtk2.8 --noconfirm

The installed files can be in folders with the suffix -2.8.
Then, check if the executable is wx-config-2.8.
If so, create the following symbolic link:

$ cd /usr/bin
$ sudo ln -s wx-config-2.8 wx-config

Another option is to download its source codes on Widgets website and compile them.

export LC_ALL="en_US.UTF-8"
```

