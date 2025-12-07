CYIFT_DIR=$(NEWIFT_DIR)/cyift
PIP_CYIFT_DIR=/usr/local/lib/python2.7/dist-packages/cyift/

# for not compiling IFT, use: make IFT=0
# for not installing cyift, use: make CYIFT=0
CYIFT=1
IFT=1

all:
# if complie ift
ifeq ($(IFT), 1)
	@echo "Compiling IFT"
	cd $(NEWIFT_DIR); $(MAKE); cd -;
endif

# if install cyift
ifeq ($(CYIFT), 1)
	@echo "Installing cpyIFT via PIP"
	cd $(CYIFT_DIR); $(MAKE); cd -;
endif
	# Compiling Cython User Specific --> -b is the output for *.so
	python setup.py build_ext -b neg_mining
	# @make clean2

clean1:
	rm -rf **/*.h **/*.c **/*.pyc **/*.so build

clean2:
	rm -rf build **/**/*.h **/**/*.c **/**/*.pyc **/*.pyc # remove all *.h, and *.c from 3 levels of subdirs
