from fastai.vision.all import *
from fastai.vision.all import cnn_learner #imports all necessary CNN libaries

from fastaudio.core.all import *
from fastaudio.augment.all import *
from fastaudio.ci import skip_if_ci#imports all necessary audio dependencies 

import os#allows for the viewing and adjustment of working directories 
import json#allows for the reading and creation of jsons
import traceback#allows for error traces to be printed

#function designed to create json files
def writeToJSONFile(fileName,data):
    with open(fileName,'w') as fp:
        json.dump(data,fp)

valid = False
#takes user input for items to predict and where to store results
while valid!=True:
    source = str(input("Enter the address of the file to run predictions on:"))
    output = str(input("Enter the address to output results too:"))
    if source!="" and output!="":
        valid = True
    else:
        print("Error: Source and output file paths required")

try:#Loads learner in context of the application
    os.chdir(os.path.abspath('..\\trained_cnn\\'))
    for file in os.listdir(os.path.abspath('..\\trained_cnn\\')):
        print(file)
        if file.endswith(".pkl"):
            learn_inf = load_learner((file), cpu=True)
    os.chdir(output)
except:#Loads learner as an independent file
    for file in os.listdir(os.path.abspath('..\\trained_cnn\\')):
        print(file)
        if file.endswith(".pkl"):
            learn_inf = load_learner((file), cpu=True)
    os.chdir(output)
try:
    #Carries out a prediction on every file of the relevant file type
    for filename in os.listdir(source):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            values ={}
            audio = AudioTensor.create(source + '\\'+filename)
            pred,pred_idx,probs = learn_inf.predict(audio)
            value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
            fname = filename.split('.')[0] + '.json'
            values['prediction'] = value
            writeToJSONFile(fname,values)
            print((filename) +" " + value)
except Exception:
    traceback.print_exc()
