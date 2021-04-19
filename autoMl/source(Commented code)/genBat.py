import os#allows for the viewing and adjustment of working directories 
import json#allows for the reading and creation of jsons

#function that creates batch files for execution
def createBat(fileName,path):
    fpath = path + "\\" + fileName
    myBat = open(r'%s.bat' %fileName,'w+')
    command = '@echo off\n CALL ' + venv + '\Scripts\\activate.bat\n ..\auto-ml-env\Scripts\python.exe ' + fpath + ".cpython-38.pyc"
    myBat.write(command)
    myBat.close()
    
with open("experiment.json", "r") as read_file:
    data = json.load(read_file)
    venv = data['venv']
    train_cnn = data['train_cnn']
    train_tab = data['train_tab']
    trained_cnn = data['trained_cnn']
    trained_tab = data['trained_tab']
    
    createBat('train_cnn',train_cnn)
    createBat('train_tab',train_tab)
    createBat('predict_cnn',trained_cnn)
    createBat('predict_tab',trained_tab)
