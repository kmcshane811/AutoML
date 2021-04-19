import json#library to allow for json manipulation
import os#library to view and alter working directories

#function designed to create json files
def writeToJSONFile(fileName,data):
    with open(fileName,'w') as fp:
        json.dump(data,fp)

data = {}
data['train'] = '..\\datasets\\train'
data['test'] = '..\\datasets\\test'
data['train_cnn'] ='..\\training_cnn'
data['train_tab'] ='..\\training_tabular'
data['trained_cnn'] ='..\\trained_cnn'
data['trained_tab'] = '..\\trained_tabular'
data['venv'] = '..\\auto-ml-env'

writeToJSONFile('experiment.json',data)#creates json that holds relative file paths
