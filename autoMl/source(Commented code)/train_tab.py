from fastaudio.core.all import *
from fastaudio.augment.all import *
from fastaudio.ci import skip_if_ci #imports all libraries required for audio manipulation

import librosa #library needed to extract audio metadata
import pandas as pd#library for creation and manipulation of dataframes
import numpy as np#allows for the creation of arrays

import os#library used for access and adjusting working directories

import json #library used for json creation and manipulation
import traceback#library used to print out error traces

from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype #imports relevant pandas types

from fastai.tabular.all import * #imports all necessary functions to create a tabular learner

#sets pandas boundaries
pd.options.display.max_rows = 20
pd.options.display.max_columns = 8

#function for creating json files
def writeToJSONFile(fileName,data):
    with open(fileName,'w') as fp:
        json.dump(data,fp)

#function for creating a tabular model
def run_learner():
    # epochs are a bit longer due to the chosen melspectrogram settings
    learn.fine_tune(1000,cbs = CSVLogger())
    out  = pd.read_csv('history.csv')
    values = {}
    results = [] 
    results.append(out.to_string())
    values['results'] = results
    writeToJSONFile('results.json',values) #generates a results json file


try:        
    with open("..\experiments\experiment.json", "r") as read_file:
        data = json.load(read_file)
        path = data['train']
        
        for file in os.listdir(os.path.abspath(path)):#identifies and loads metadata file
            if file.endswith(".csv") and file!='dataset.csv':
                df = pd.read_csv(path+"\\"+file)
                break
            
        label = list(df['label']) #fetches the list of labels 

        #defines a header for the processed dataframe
        header = 'filename chroma_stft rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
        for i in range(1, 21):
            header += f' mfcc{i}'
        header += ' label'
        header = header.split()
        
        dfPath = path + '\\dataset.csv'
        tabFile = open(os.path.abspath(dfPath), 'w', newline='')

        with tabFile:
            writer = csv.writer(tabFile)
            writer.writerow(header)

            #takes each file of the relevant type for processing
            for filename in os.listdir(os.path.abspath(path)):
                songname = filename
                if songname.lower().endswith(('.mp3','.wav')):

                    if songname.find("unlabelled")==-1:
                        #converts each audio file into the relevant meta data
                        y, sr = librosa.load(path +'\\' +songname, mono=True, duration=30)
                        rmse = librosa.feature.rms(y=y)[0]
                        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
                        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
                        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
                        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
                        zcr = librosa.feature.zero_crossing_rate(y)
                        mfcc = librosa.feature.mfcc(y=y, sr=sr)
                        label = list(df[df.filename == filename]['label'])

                        #appends data to csv
                        to_append = f'{filename} {np.mean(chroma_stft)} {np.mean(rmse)} {np.mean(spec_cent)} {np.mean(spec_bw)} {np.mean(rolloff)} {np.mean(zcr)}'
                      
                        for e in mfcc:
                            to_append += f' {np.mean(e)}'
                        
                        to_append += f' {label[0]}'
                        tabFile = open(os.path.abspath(dfPath), 'a', newline='')

                        with tabFile:
                            writer = csv.writer(tabFile)
                            writer.writerow(to_append.split())
        tabFile.close()

        #loads dataframe
        df_nn = pd.read_csv(os.path.abspath(dfPath),low_memory= False)
        df_nn = df_nn.iloc[1:]
        
        dep_var = 'label'
        
        procs_nn = [Categorify, FillMissing, Normalize]
        cont_nn,cat_nn = cont_cat_split(df_nn, max_card=9000, dep_var=dep_var)

        to_nn = TabularPandas(df_nn, procs_nn, cont_nn, y_names=dep_var,splits =RandomSplitter(valid_pct = 0.4, seed = 42)(range_of(df_nn))) #defines pandas datablock

        dls = to_nn.dataloaders(64, num_workers=0) #converts datablock into data loader

        learn = tabular_learner(dls, metrics=accuracy, layers=[500,200]) # defines tabular learner 

        os.chdir(data['trained_tab'])
        # We only validate the model when running in CI
        run_learner() #trains model
        learn.export()#exports trained model
except Exception:
    traceback.print_exc()
