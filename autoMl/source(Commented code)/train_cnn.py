from fastai.vision.all import *
from fastai.vision.all import cnn_learner #imports relevant libraries to create a CNN model

from fastaudio.core.all import *
from fastaudio.augment.all import *
from fastaudio.ci import skip_if_ci       #imports relevant libraries to prepare audio data

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype #imports library for dataframe manipulation
import json #imports library for JSON manipulation
import os   #imports library for directory control
import traceback #library to produce error traces

#function designed to create a JSON file
def writeToJSONFile(fileName,data):
    with open(fileName,'w') as fp:
        json.dump(data,fp)
        
@skip_if_ci
#function designed to train a cnn model
def run_learner():
    # epochs are a bit longer due to the chosen melspectrogram settings
    learn.fine_tune(10,cbs = CSVLogger())
    out  = pd.read_csv('history.csv')
    print(out.to_string())
    values = {}
    results = [] 
    print(type(out.to_string()))
    results.append(out.to_string())
    values['results'] = results
    writeToJSONFile('results.json',values) #creates a results json file
    
  

 

        
try:        
    with open("..\experiments\experiment.json", "r") as read_file:
        data = json.load(read_file)
        path = data['train'] #fetches necessary file paths
        for file in os.listdir(os.path.abspath(path)):
            if file.endswith(".csv") and file!='dataset.csv':
                df = pd.read_csv(path+"\\"+file) #initialises dataframe
                break
        #identifies files and their labels    
        x = str('filename')
        y = str('label')
        z = path + '\\'

        #Describes necessary data preprocessing for training 
        cfg = AudioConfig.BasicMelSpectrogram(n_fft=512)
        a2s = AudioToSpec.from_cfg(cfg)
        item_tfms = [ResizeSignal(4500),a2s]

        #initialises a data block from the dataset
        auds = DataBlock(blocks = (AudioBlock, CategoryBlock), 
                         get_x = ColReader(x, pref=z), 
                         splitter = TrainTestSplitter(test_size=0.2, random_state=42, stratify=None, shuffle=True),
                         item_tfms = item_tfms,
                         get_y = ColReader(y))

        #converts data block into a data loader
        dbunch = auds.dataloaders(df, bs=32,num_workers=0)
        learn = cnn_learner(dbunch, 
                    resnet18,
                    n_in=1,  # <- This is the only audio specific modification here
                    loss_func=CrossEntropyLossFlat(),
                    metrics=[accuracy,error_rate])

        os.chdir(data['trained_cnn'])#changes directory for storing results and exported model
        
        run_learner() #trains the cnn
        learn.export()#outputs the trained model as a .pkl file
except Exception:
    traceback.print_exc()
