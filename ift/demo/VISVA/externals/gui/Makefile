all: libgui

LIB=.
HDR=./include
SRC=./src
OBJ=./obj


#Rules
libgui: $(LIB)/libgui.a
	echo "libgui.a built..."

$(LIB)/libgui.a: \
$(OBJ)/guibasic.o \
$(OBJ)/plotcurve.o \
$(OBJ)/cursor.o \
$(OBJ)/colourbutton.o \
$(OBJ)/bitmaptogglebutton.o \
$(OBJ)/bricontrdialog.o \
$(OBJ)/brushpicker.o \
$(OBJ)/canvas.o \
$(OBJ)/rendercanvas.o \
$(OBJ)/chartwindow.o \
$(OBJ)/simplecanvas.o \
$(OBJ)/basepanel.o \
$(OBJ)/basedialog.o \
$(OBJ)/spinslider.o \
$(OBJ)/interactionhandler.o \
$(OBJ)/labeledbitmapbutton.o \
$(OBJ)/alphacolourbutton.o \
$(OBJ)/bitmapradiobutton.o \
$(OBJ)/floatctrl.o \
$(OBJ)/spinfloat.o
	ar csr $(LIB)/libgui.a \
$(OBJ)/guibasic.o \
$(OBJ)/plotcurve.o \
$(OBJ)/cursor.o \
$(OBJ)/colourbutton.o \
$(OBJ)/bitmaptogglebutton.o \
$(OBJ)/bricontrdialog.o \
$(OBJ)/brushpicker.o \
$(OBJ)/canvas.o \
$(OBJ)/rendercanvas.o \
$(OBJ)/chartwindow.o \
$(OBJ)/simplecanvas.o \
$(OBJ)/basepanel.o \
$(OBJ)/basedialog.o \
$(OBJ)/spinslider.o \
$(OBJ)/interactionhandler.o \
$(OBJ)/labeledbitmapbutton.o \
$(OBJ)/alphacolourbutton.o \
$(OBJ)/bitmapradiobutton.o \
$(OBJ)/floatctrl.o \
$(OBJ)/spinfloat.o


$(OBJ)/guibasic.o: $(SRC)/guibasic.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/guibasic.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/guibasic.o

$(OBJ)/plotcurve.o: $(SRC)/plotcurve.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/plotcurve.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/plotcurve.o

$(OBJ)/cursor.o: $(SRC)/cursor.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/cursor.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/cursor.o

$(OBJ)/colourbutton.o: $(SRC)/colourbutton.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/colourbutton.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/colourbutton.o

$(OBJ)/bitmaptogglebutton.o: $(SRC)/bitmaptogglebutton.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/bitmaptogglebutton.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/bitmaptogglebutton.o

$(OBJ)/bricontrdialog.o: $(SRC)/bricontrdialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/bricontrdialog.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/bricontrdialog.o

$(OBJ)/brushpicker.o: $(SRC)/brushpicker.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/brushpicker.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/brushpicker.o

$(OBJ)/canvas.o: $(SRC)/canvas.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/canvas.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/canvas.o

$(OBJ)/rendercanvas.o: $(SRC)/rendercanvas.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/rendercanvas.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/rendercanvas.o

$(OBJ)/chartwindow.o: $(SRC)/chartwindow.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/chartwindow.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/chartwindow.o

$(OBJ)/simplecanvas.o: $(SRC)/simplecanvas.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/simplecanvas.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/simplecanvas.o

$(OBJ)/basepanel.o: $(SRC)/basepanel.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/basepanel.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/basepanel.o

$(OBJ)/basedialog.o: $(SRC)/basedialog.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/basedialog.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/basedialog.o

$(OBJ)/spinslider.o: $(SRC)/spinslider.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/spinslider.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/spinslider.o

$(OBJ)/interactionhandler.o: $(SRC)/interactionhandler.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/interactionhandler.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/interactionhandler.o

$(OBJ)/labeledbitmapbutton.o: $(SRC)/labeledbitmapbutton.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/labeledbitmapbutton.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/labeledbitmapbutton.o

$(OBJ)/alphacolourbutton.o: $(SRC)/alphacolourbutton.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/alphacolourbutton.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/alphacolourbutton.o

$(OBJ)/bitmapradiobutton.o: $(SRC)/bitmapradiobutton.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/bitmapradiobutton.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/bitmapradiobutton.o

$(OBJ)/floatctrl.o: $(SRC)/floatctrl.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/floatctrl.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/floatctrl.o

$(OBJ)/spinfloat.o: $(SRC)/spinfloat.cpp
	$(CXX) $(CC_FLAGS) $(OLDIFT_INCLUDE) $(OPF_INCLUDE) -c $(SRC)/spinfloat.cpp \
	-I$(HDR) $(WXFLAGS) -o $(OBJ)/spinfloat.o

clean:
	$(RM) $(LIB)/lib*.a -f; $(RM) $(OBJ)/*.o *~ -f

