from experiments.test.LSTM_test import test_model as LSTMTest
from experiments.test.LTCN_test import test_model as LTCNTest
from experiments.test.QRoPET_test import test_model as QRoPETTest
from experiments.test.QATN_test import test_model as QATNTest

import sys
import traceback
import os

if __name__ == "__main__":
    try:
        model = sys.argv[1]
        print(model)

        if model == "lstm":
            LSTMTest()
        elif model == "ltcn":
            LTCNTest()
        elif model == "qropet":
            QRoPETTest()
        elif model == "qatn":
            QATNTest()
        else:
            print("no model selected")
    
    except Exception as e:
        import jax 
        if jax.process_index() == 0:
            traceback.print_exc()
