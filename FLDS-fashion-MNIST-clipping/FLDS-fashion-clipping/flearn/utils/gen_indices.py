import hashlib
import numpy as np

d = 5 # number of counter arrays
#m = [100, 200, 1000, 2000, 800, 1600]
#model_lens = [217434, 51930, 36720]
m = 1600
model_lens = [36720]

for model_len in model_lens:
    outfile = "./hash_indices/shakespeare_stacked_lstm_5_1600"
    hash_indices = np.zeros((model_len, d, 2))  # d*(position, sign) for each dimension
    for i in range(model_len):
        sha256 = hashlib.sha256(str(hash(i)).encode())
        sha256_sign = hashlib.sha256(str(hash(i)).encode())
        for j in range(d):
            sha256.update(str(hash(j)).encode())
            sha256_sign.update(str(hash(i*2)).encode())
            position = int(sha256.hexdigest(), 16) % m
            sign = (int(sha256_sign.hexdigest(), 16) % 2) * 2 - 1
            hash_indices[i][j][0], hash_indices[i][j][1] = position, sign

    np.save(outfile, hash_indices)




