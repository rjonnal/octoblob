import json

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)

class Parameters(dict):
    """Implement a python dictionary that has persistent json storage as a record
    of processing and data analysis."""
    def __init__(self, filename, verbose=False):
        self.filename = filename
        
        self.verbose = verbose
        try:
            temp = self.load()
            for k in temp.keys():
                self[k] = temp[k]
        except Exception as e:
            print(e)
            pass
                
    def __getitem__(self, key):
        val = dict.__getitem__(self, key)
        if self.verbose:
            print('GET', key)
        return val

    def __setitem__(self, key, val):
        if self.verbose:
            print('SET', key, val)
        dict.__setitem__(self, key, val)
        self.save()
        
    def __repr__(self):
        dictrepr = dict.__repr__(self)
        return '%s(%s)' % (type(self).__name__, dictrepr)

    def clear(self):
        keys = list(self.keys())
        for k in keys:
            dict.pop(self,k)
        self.save()
        
        
    def get_param_filename(self,filename):
        outfile = filename.replace('.unp','')+'_parameters.json'
        return outfile

    def load(self):
        # load a json file into a dictionary
        with open(self.filename,'r') as fid:
            dstr = fid.read()
            dictionary = json.loads(dstr)
        return dictionary

    def save(self):
        temp = {}
        for k in self.keys():
            temp[k] = self[k]
        dstr = json.dumps(temp,indent=4, sort_keys=True, cls=NpEncoder)
        with open(self.filename,'w') as fid:
            fid.write(dstr)

