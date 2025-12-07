CYIFT_DIR=$(NEWIFT_DIR)/cyift/
PIP_CYIFT_DIR=/usr/local/lib/python2.7/dist-packages/cyift/

all: uninstall install

clean: uninstall
	rm -rf ./cyift/*.so build cyift/*.c cyift/*.h

	
install:
	@echo "Installing cpyIFT via PIP"
	sudo pip install $(CYIFT_DIR)

uninstall:
	if [ -d $(PIP_CYIFT_DIR) ]; then \
		echo "\n---> Uninstalling CpyIFT Package"; \
		sudo pip uninstall cyift; \
	fi