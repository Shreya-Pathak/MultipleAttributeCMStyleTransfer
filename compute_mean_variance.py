import os
import sys
import json
import numpy as np

dr = sys.argv[1]
l=[]
for f1 in os.listdir(dr):
    x = os.path.join(os.path.join(dr,f1),"all_results.json")
    if os.path.isfile(x):
        l=l+[json.load(open(x))['predict_perplexity']]
print(l)
arr = np.array(l)
print("Mean: ", np.mean(arr))
print("Variance: ", np.var(arr))
print("Standard deviation: ", np.std(arr))
