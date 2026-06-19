import soundata

dataset = soundata.initialize("urbansound8k", data_home="training/data/urbansound8k")
dataset.download()
dataset.validate()
