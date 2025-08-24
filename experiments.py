from experiments.LSTM_forecast import train_model as LSTMTrain
from experiments.LTCN_forecast import train_model as LTCNTrain
from experiments.QATN_forecast import train_model as QATNTrain

import sys
if __name__ == "__main__":
    model = sys.argv[1]
    
    if model == "lstm":
        LSTMTrain()
    elif model == "ltcn":
        LTCNTrain()
    elif model == "qatn":
        QATNTrain()
    else:
        print("no model selected")