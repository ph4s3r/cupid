##########################################################################################################
# Author: Peter Karacsonyi                                                                               #
# Last updated: 2024 feb 8                                                                               #
# runs tile extraction for each wsi specified in a folder, as a subprocess                               #
##########################################################################################################


##########
# CONFIG #
##########
config_json_path = "/mnt/bigdata/cupid/preprocessing/tileextractor_conf.json"

###########
# imports #
###########
import os
import json
import shutil
import psutil
import subprocess
from colorama import init, Fore # https://github.com/tartley/colorama/blob/master/colorama/ansi.py
from pathlib import Path

init(autoreset=True)

config = {}

try:
    assert os.path.isfile(config_json_path), f"file not found {config_json_path}"
    with open(config_json_path, 'r') as f:
        config = json.loads(f.read())
except Exception as e:
    print(f"Could not read config file {config_json_path}")
    print(f"Error: {e}")
    os._exit(1)


print("")
print(f"**************************************************")
print(f"***************** TILE EXTRACTOR *****************")
print(f"**************************************************")
print("")

print("CONFIG:")
print(json.dumps(config, indent=4))
Path(config.get('wsi_done_folder')).mkdir(parents=True, exist_ok=True)

swap_avail = int(psutil.swap_memory().total)
if swap_avail < int(psutil.virtual_memory().available):
    print(Fore.RED + f"WARNING: only ({swap_avail} MB) swap is available which is less than 2x of the physical memory. The OS will terminate the process when an image does not fit into the memory.")
    print("")

####################################
# run for all slides in the folder #
####################################

wsi_paths = list(Path(config.get('wsi_folder')).glob("*.tif*"))
for wsi_path in wsi_paths:
    
    command = ['python', 'preprocessing/TileExtractor-Process-WSI.py', wsi_path, config_json_path, config.get('out_folder')]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # streaming subprocess stdout
    while True:
        output_line = process.stdout.readline()
        if output_line == '' and process.poll() is not None:
            break
        if output_line:
            print("\t", output_line.strip(), flush=True)
    
    exit_code = process.wait()
    if exit_code == 0:
        newloc = config.get('wsi_done_folder') + "/" + str(wsi_path.name)
        try:
            shutil.move(str(wsi_path), newloc)
        except Exception as e:
            print(Fore.YELLOW + f"error moving wsi to {newloc}, leaving where it was..")
    elif exit_code == 78:
        print(Fore.RED + f"exiting runner due to config read error.")
        os._exit(1)
    else:
        stderr_output = process.stderr.read().strip()
        if stderr_output:
            print(Fore.RED + f"error in subprocess for {wsi_path.name}: {stderr_output}. skipping to next slide")
        else:
            print(Fore.RED + f"processing of {wsi_path} failed with exit code: {exit_code}. skipping to next slide")    

    print("\r\n")
    
