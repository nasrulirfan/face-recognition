# How to setup
## Windows specific installation
1. Clone the repo

```bash
git clone https://github.com/nasrulirfan/face-recognition.git
```
#### You can install git with Windows, check at https://github.com/git-for-windows/git/releases/download/v2.44.0.windows.1/Git-2.44.0-64-bit.exe

2. Install tensorflow, please refer to this link https://www.tensorflow.org/install/pip#step-by-step_instructions
- Please also make sure you create a conda environment ONLY with Python 3.9

3. Enter into conda environment (it should be installed when you follow step 3 inside the tensorflow instruction)
```bash
conda activate <your conda environment name>
```
- In case you haven't installed yet when following tensorflow instruction, the miniconda download link for Windows is https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe

4. Make sure directory is inside project already, if not just 
```bash
cd <path to your project>
```

5. Install all requirements using pip
```bash
pip install -r requirements.txt
```

6. Install MongoDB (if not installed yet) https://www.mongodb.com/try/download/community and run your MongoDB

7. Check at application.py (line 32 `client = pymongo.MongoClient("mongodb://localhost:<your port>/<your database>")`) and update it with your own mongodb port and database name

8. Run your flask application
```bash
python application.py
```
9. Open your browser and go to http://localhost:9874/ to enter your flask application

# Validation
The 3 models validation jupyter notebook are in 'validation' folder 



