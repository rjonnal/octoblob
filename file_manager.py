import os
import pathlib
import shutil

def get_params_filename(data_filename):
    params_filename = os.path.join(os.path.split(data_filename)[0],'processing_parameters.json')
    return params_filename


def get_bscan_folder(data_filename,make=True):
    bscan_folder = data_filename.replace('.unp','')+'_bscans'
    
    if make:
        os.makedirs(bscan_folder,exist_ok=True)
    return bscan_folder


def get_org_folder(data_filename,make=True):
    bscan_folder = data_filename.replace('.unp','')+'_org'
    
    if make:
        os.makedirs(bscan_folder,exist_ok=True)
    return bscan_folder

bscan_template = 'complex_bscan_%05d.npy'


def cleanup_folders(folder_filters=[],delete=False):
    folders = []
    for ff in folder_filters:
        temp = list(pathlib.Path('.').rglob(ff))
        for item in temp:
            if os.path.isdir(item):
                folders.append(item)

    for f in folders:
        if not delete:
            print('Would delete %s.'%f)
        
        else:
            print('Deleting %s.'%f)
            shutil.rmtree(f)


def cleanup_files(file_filters=[],delete=False):
    files = []
    for ff in file_filters:
        temp = list(pathlib.Path('.').rglob(ff))
        for item in temp:
            if os.path.isfile(item):
                files.append(item)

    for f in files:
        if not delete:
            print('Would delete %s.'%f)
        
        else:
            print('Deleting %s.'%f)
            os.remove(f)

def clean(delete=False):
    cleanup_folders(['*_diagnostics','*_bscans','*_org'],delete=delete)
    cleanup_files(['processing_parameters.json'],delete=delete)
    cleanup_files(['octoblob.log'],delete=delete)
    
