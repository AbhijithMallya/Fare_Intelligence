from flask import Flask , render_template , request
import pickle
import numpy as np
app = Flask(__name__ , static_url_path='/static')

meaning = { 'True':True,
            'False':False,
            '1a' : [True,False,False,False,False,False],
            '2a' : [False,True,False,False,False,False],
            '2s' : [False,False,True,False,False,False],
            '3a' : [False,False,False,True,False,False],
            'cc' : [False,False,False,False,True,False],
            'sl' : [False,False,False,False,False,True],
             }


@app.route("/" , methods=['GET','POST'])
def home():
    final_output = 5
    if request.method == 'POST':
        # Process the form data when the form is submitted
        catering_service = request.form.get('CateringService')
        dynamic_fare = request.form.get('DynamicFare')
        distance = request.form.get('distance')
        duration = request.form.get('duration')
        train_class= request.form.get('options')

        # Here, you can perform any logic to handle the form data
        # For now, let's print the values to the console
        print(f'Catering Service: {catering_service}')
        print(f'Dynamic Fare: {dynamic_fare}')
        print(f'Distance: {distance}')
        print(f'Duration: {duration}')
        print(f'Train Class: {train_class}')

        model = pickle.load(open('GradientBoostingRegressor.pkl', 'rb'))
        print("Model loaded successfully")

        # input = [catering_service , dynamic_fare] + meaning[selected_option]
        input = [meaning[catering_service],meaning[dynamic_fare],float(distance),float(duration)]+ meaning[train_class]
        print("Imput Array : ",input)
        # input_features = np.array([False, False, 1255, 160, True, False, False, False, False, False]) answer 3844.7819
        input_features = np.array(input)
        input_features_reshaped = input_features.reshape(1, -1)
        result = model.predict(input_features_reshaped)
        print( "Predicted Fare : ",result)
        result = str("The Predicted Fare :: "+ str(result))
        
        return render_template('index.html',span =result )


   
    return render_template('index.html')

#Development Server
if __name__ == '__main__':
   app.run(debug=True)


#Production Server
# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000 )    

