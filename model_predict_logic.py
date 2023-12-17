import numpy as np
import pickle
#1A,2A,2S,3A,CC,SL
meaning = { 'True':True,
            'False':False,
            '1a' : [True,False,False,False,False,False],
            '2a' : [False,True,False,False,False,False],
            '2s' : [False,False,True,False,False,False],
            '3a' : [False,False,False,True,False,False],
            'cc' : [False,False,False,False,True,False],
            'sl' : [False,False,False,False,False,True],
             }
model = pickle.load(open('GradientBoostingRegressor.pkl', 'rb'))
catering_service='False'
dynamic_fare = 'False'
distance = 1255
duration = 160
train_class = '1a'

input = [meaning[catering_service],meaning[dynamic_fare],distance,duration]+ meaning[train_class]

input_features = np.array(input)
input_features_reshaped = input_features.reshape(1, -1)
result = model.predict(input_features_reshaped)
print("a1 : ",input)
print("result : ",result)
print("Code sucess")

