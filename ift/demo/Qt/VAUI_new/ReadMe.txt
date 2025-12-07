**Compile and Run**
#0. open the terminal

#1. go to the project folder
cd some/path/to/ift/folder/demo/Qt/VAUI_new

#2.generate makefile
qmake Teste.pro

#3. compile
make

#4. run
./Teste


**Dependecnies**
Qt 5.7 or higher (https://info.qt.io/download-qt-for-application-development)


**Commom Issues**

*qmake is not defnied
After downloading Qt, you can find qmake on /home/user/Qt/x.x/gcc_64/bin/qmake. So the second step in "Compile and Run"
section becomes

/home/user/Qt/x.x/gcc_64/bin/qmake Teste.pro

where x.x is the Qt version (e.g. 5.8)
