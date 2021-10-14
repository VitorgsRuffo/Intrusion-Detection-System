import pandas as pd
import numpy as np
import os

data_frame = pd.read_csv(os.path.join('../', 'training_day_1.csv'), dtype={"bytes": "Int32", "packets": "Int32", "label": "Int32"})

bytes_count = np.array(data_frame.bytes).reshape(-1, 1)
print(bytes_count)
