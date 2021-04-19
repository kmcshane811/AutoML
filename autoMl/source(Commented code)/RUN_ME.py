import subprocess    #library to execute batch files
import tkinter as tk #library to initialise GUI
import json          #library to allow for json manipulation
import os            #library to view and alter working directories
import traceback     #library to produce error traces
from subprocess import run

#function to call the cnn trainer
def start_batch1():
           subprocess.call([r'train_cnn.bat'])

#function to call tabular trainer
def start_batch2(): 
           subprocess.call([r'train_tab.bat'])

#function to carry out cnn predictions
def start_batch3(): 
           subprocess.call([r'predict_cnn.bat'])

#function to carry out tabular functions (not yet functional)
def start_batch4(): 
           subprocess.call([r'predict_tab.bat'])

#function to output the results of the last trained cnn model
def start_batch5():
    subprocess.call([r'results_cnn.bat'])

#function to output the results of the last trained tabular model
def start_batch6():
    subprocess.call([r'results_tab.bat'])

#function to install all necessary requirements to a VENV
def pip_install():
    subprocess.call('../auto-ml-env/Scripts/pip install librosa==0.8.0')
    subprocess.call('../auto-ml-env/Scripts/pip install pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio===0.8.0 -f https://download.pytorch.org/whl/torch_stable.html')
    subprocess.call('../auto-ml-env/Scripts/pip install fastai==2.2.7')
    subprocess.call('../auto-ml-env/Scripts/pip install fastaudio==0.1.4')
    subprocess.call('../auto-ml-env/Scripts/pip install -Uqq fastbook')
    subprocess.call('../auto-ml-env/Scripts/pip install scikit-build')
            
path = '.\\experiments\\' #identifies the directory path to the batch files

#Checks for the existence of a VENV and creates a new venv if needed
try:
    cwd = os.getcwd()
    venv = False
    for file in os.listdir(os.getcwd()):
        if file == "auto-ml-env":
            venv = True
    print(venv)
    if venv == False:
        os.chdir(os.path.abspath(path)) #changes directory to experiments
        subprocess.call([r'create_venv.bat'])
        pip_install()
        os.chdir(os.path.abspath(cwd)) #returns directory to the main folder
except Exception:
    traceback.print_exc()
    
        
    
os.chdir(os.path.abspath(path)) #changes directory to experiments

#Initialises the GUI for end-user use
root= tk.Tk()

button1 = tk.Button (root, text='Train Audio CNN ',command=start_batch1,bg='green',fg='white')
button2 = tk.Button (root, text='Train Tabular Learner ',command=start_batch2,bg='green',fg='white')
button3 = tk.Button (root, text='Run CNN Predictions ',command=start_batch3,bg='green',fg='white')
button4 = tk.Button (root, text='Run Tabular Predictions ',command=start_batch4,bg='green',fg='white')
button5 = tk.Button (root, text='Show CNN results ',command=start_batch5,bg='green',fg='white')
button6 = tk.Button (root, text='Show Tabular Results ',command=start_batch6,bg='green',fg='white')

button1.place(x=50,y=20)
button2.place(x=50,y=40)
button3.place(x=50,y=60)
button4.place(x=50,y=80)
button5.place(x=50,y=100)
button6.place(x=50,y=120)
                       
root.mainloop()
