from main import np,ozone_data



train_size = int(0.8 * len(ozone_data))
train_data, test_data = ozone_data[:train_size], ozone_data[train_size:]

train_mean = np.mean(train_data)
train_std = np.std(train_data)
train_data = (train_data - train_mean) / train_std


test_mean = np.mean(test_data)
test_std = np.std(test_data)
test_data = (test_data - test_mean) / test_std
