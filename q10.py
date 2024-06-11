import numpy as np
data = np.array([['a', 1, 5],
                 ['b', 2, 7],
                 [np.nan, np.nan, 8],
                 ['a', 4, np.nan],
                 ['b', 5, 3]])
for i in range(data.shape[1]):
    col = data[:, i]
    col_missing = np.where(col == 'nan')[0]
    if col_missing.size > 0:
        col = np.delete(col, col_missing)
        try:
            col = col.astype(float)
        except ValueError:
            unique_values, indices = np.unique(col, return_inverse=True)
            col = indices.astype(float)
        col_mean = np.mean(col)
        data[col_missing, i] = str(col_mean)

unique_values, indices = np.unique(data[:, 0], return_inverse=True)
one_hot_encoded = np.eye(len(unique_values))[indices]
encoded_data = np.hstack((one_hot_encoded, data[:, 1:].astype(float)))

print("Processed Data:")
print(encoded_data)


# OUTPUT
# Processed Data:
# [[0.   1.   0.   1.   5.  ]
#  [0.   0.   1.   2.   7.  ]
#  [1.   0.   0.   3.   8.  ]
#  [0.   1.   0.   4.   5.75]
#  [0.   0.   1.   5.   3.  ]]