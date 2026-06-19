import urllib.request  # workaround del bug de soundata con urllib
import soundata

dataset = soundata.initialize("urbansound8k", data_home="training/data/urbansound8k")
dataset.download()
dataset.validate()