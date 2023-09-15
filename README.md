## pyqt tools

### How to make

0. prerequisite
	- conda
	- `pip install pyqt5`
	- `pip install qt5-tools` for desiginer
	- `pip install python-opencv` & `conda install opencv`
	- `pip install pyinstaller` for pyinstaller

1. pyqt designer
	```
	pyqt5-tools designer
	```
2. pyqt code write
	- using vscode, write a code

3. pyinstaller
	```
	pyinstaller [py file]
	# -w : no window
	# -F, --onefile : single file
	```

### TODO
- pyinstaller is heavy
	- bandicutcut : 302 MB
	- because of OpenCV?
- using .ui file
	- bandicutcut.exe require .ui file too
