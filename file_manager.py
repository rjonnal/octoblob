import os

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
