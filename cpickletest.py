import pickle

with open("testpickle.dev",'rb') as f:
    y = pickle.load(f)
    print y

