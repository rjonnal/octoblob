from octoblob import data_browser
import sys

try:
    file_filters = sys.argv[1:]
except:
    file_filters=['*.npy','*.png']

b = data_browser.Browser()
b.browse(root='.',file_filters=file_filters)
