import sys
from flask import Flask , render_template ,request
from src.pipelines.predict_pipeline import CustomData, PredictPipeline
from src.exception import CustomException


app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict" , methods = [ "GET","POST" ] )
def predict_datapoint():
    if request.method == "GET":
        return render_template("predict.html")
    else:
        try:
            user_data = CustomData(
                buying = request.form.get("buying") ,
                maint= request.form.get("maint"),
                doors= request.form.get("doors"),
                persons= request.form.get("persons"),
                lug_boot= request.form.get("lug_boot"),
                safety= request.form.get("safety")
            )
            
            user_data_df = user_data.get_data_as_dataframe()

            predict_pipeline = PredictPipeline()
            preds = predict_pipeline.predict(user_data_df)
            
            return render_template( "predict.html", results = preds[0][0] )
        except Exception as e:
            raise CustomException(e , sys)
            
if __name__ == "__main__":
    app.run(host = "0.0.0.0", port = 8080 , debug  = True)