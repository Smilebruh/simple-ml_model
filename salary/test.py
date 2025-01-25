import pickle
import numpy as np

with open("../dataset/salary/my_model.pkl",'rb') as file:
    model = pickle.load(file)
    file.close()

experience = np.array([float(input("input years of experience of your job : "))]).reshape(-1,1)

print(f"here's your salary {model.predict(experience)[0,0] : .2f}$")