import os#allows for the viewing and adjustment of working directories 
import json#allows for the reading and creation of jsons
import traceback#allows for error traces to be printed

try:
    try:#returns results in context of the application 
        with open("experiment.json","r") as path_file:
            data = json.load(path_file)
            os.chdir(os.path.abspath(data['trained_tab']))
            with open("results.json", "r") as read_file:
                values = json.load(read_file)
                results = values['results'][0]
                print(results)
    except:#returns results as an independent file
        with open("results.json", "r") as read_file:
                values = json.load(read_file)
                results = values['results'][0]
                print(results)
except Exception:
    traceback.print_exc()
