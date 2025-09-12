from experiments.train.LSTM_train import train_model as LSTMTrain
from experiments.train.LTCN_train import train_model as LTCNTrain
from experiments.train.QRoPET_train import train_model as QRoPETTrain
from experiments.train.QATN_train import train_model as QATNTrain

import sys
import traceback
import os

if __name__ == "__main__":
    try:
        model = sys.argv[1]
        print(model)
        if model == "lstm":
            LSTMTrain()
        elif model == "ltcn":
            LTCNTrain()
        elif model == "qropet":
            QRoPETTrain()
        elif model == "qatn":
            QATNTrain()
        else:
            print("no model selected")
    
    except Exception as e:
        import jax 
        if jax.process_index() == 0:
            traceback.print_exc()

