import os,sys
from datetime import datetime
import logging
from xml.etree import ElementTree as ET


XML_DICT = {}
# populate XML_DICT with required parameters from Yifan's XML grammar
# keys of this dictionary [x,y] are x = element tag and y = element attribute
# the values of this dictionary (x,y) are x = our new name for the data and
# y = the data type (i.e. a function that we can cast the output with)
XML_DICT['Time','Data_Acquired_at'] = ('time_stamp',str)
XML_DICT['Volume_Size','Width'] = ('n_depth',int)
XML_DICT['Volume_Size','Height'] = ('n_fast',int)
XML_DICT['Volume_Size','Number_of_Frames'] = ('n_slow',int)
XML_DICT['Volume_Size','Number_of_Volumes'] = ('n_vol',int)
XML_DICT['Scanning_Parameters','X_Scan_Range'] = ('x_scan_mv',int)
XML_DICT['Scanning_Parameters','X_Scan_Offset'] = ('x_offset_mv',int)
XML_DICT['Scanning_Parameters','Y_Scan_Range'] = ('y_scan_mv',int)
XML_DICT['Scanning_Parameters','Y_Scan_Offset'] = ('y_offset_mv',int)
XML_DICT['Scanning_Parameters','Number_of_BM_scans'] = ('n_bm_scans',int)



def get_configuration(filename):

    ''' Pull configuration parameters from Yifan's
    config file. An example configuration file is shown
    below. Calling get_configuration('temp.xml') returns
    a dictionary of parameters useful for processing the OCT
    stack, e.g. numbers of scans in x and y directions,
    voltage range of scanners, etc.

    Example XML config file:

    <?xml version="1.0" encoding="utf-8"?>
    <MonsterList>
     <!--Program Generated Easy Monster-->
     <Monster>
      <Name>Goblin</Name>
      <Time
       Data_Acquired_at="1/30/2018 12:21:22 PM" />
      <Volume_Size
       Width="2048"
       Height="400"
       Number_of_Frames="800"
       Number_of_Volumes="1" />
      <Scanning_Parameters
       X_Scan_Range="3000"
       X_Scan_Offset="0"
       Y_Scan_Range="0"
       Y_Scan_Offset="0"
       Number_of_BM_scans="2" />
      <Dispersion_Parameters
       C2="0"
       C3="0" />
     </Monster>
    </MonsterList>

    Example output dictionary:

    {'y_offset_mv': 0, 'x_offset_mv': 0, 'n_fast': 400, 
     'y_scan_mv': 0, 'n_slow': 800, 'n_vol': 1, 
     'x_scan_mv': 3000, 'time_stamp': '1/30/2018 12:21:22 PM', 
     'n_bm_scans': 2, 'n_depth': 2048}

    '''
    
    # append extension if it's not there
    if not filename[-4:].lower()=='.xml':
        filename = filename + '.xml'

    
    # use Python's ElementTree to get a navigable XML tree
    temp = ET.parse(filename).getroot()

    # start at the root, called 'Monster' for whatever reason:
    tree = temp.find('Monster')

    # make an empty output dictionary
    config_dict = {}

    # iterate through keys of specification (XML_DICT)
    # and find corresponding settings in the XML tree.
    # as they are found, insert them into config_dict with
    # some sensible but compact names, casting them as
    # necessary:
    for xml_key in XML_DICT.keys():
        node = tree.find(xml_key[0])
        config_value = node.attrib[xml_key[1]]
        xml_value = XML_DICT[xml_key]
        config_key = xml_value[0]
        config_cast = xml_value[1]
        config_dict[config_key] = config_cast(config_value)
        
    return config_dict
    
def make_configuration():

    config = {}
    for xml_value in XML_DICT.values():
        config_key = xml_value[0]
        config[config_key] = None

    return config
