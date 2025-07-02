colqwen-visualizer-py-env\Scripts\activate.ps1

# The _internal folder should contain two things:
# 1. "_models" folder that contains the hf-hub colqwen2-basic and colqwen2-1.0 models copied over
# 2. "poppler-24.08.0" folder containing the poppler library
pyinstaller .\colqwen-visualizer.py -p . --noconfirm --add-data _internal:.