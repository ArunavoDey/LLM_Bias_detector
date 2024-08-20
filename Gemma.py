import pandas as pd
import os  # For configuring Kaggle credentials
import keras
import keras_nlp
from IPython.display import Markdown  

class Gemma:
    def __init__(self):
        self.preset = "gemma_instruct_2b_en" # name of pretrained Gemma
        self.seed = 42
        self.sequence_length = 512 # max size of input sequence for training
        self.batch_size = 1 # size of the input batch in training, x 2 as two GPUs
        self.epochs = 1 # number of epochs to train
    def load(self):
        # Define and instantiate the Gemma model
        
        self.model = keras_nlp.models.GemmaCausalLM.from_preset(self.preset)

        self.model.summary()
        return self.model
    
    def generate(self, query):
        return self.model.generate(query, max_length=256)
