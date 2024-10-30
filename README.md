# practicaml-004
en este dataset voy a crear un modelo que classificara si aceptar o no aceptar la compra de un automovil en base a sus caracteristicas
url : https://www.kaggle.com/datasets/stealthtechnologies/car-evaluation-classification

rendimiento del dataset

modelo con el mejor rendimiento:XGBClassifier <br>
rendimiento del del modelo en el test set : 0.962173 <br>

inconvenientes durante la creacion del modelo: debido a que las labels , estaban en formato de string, se tuvo que crear un ordinal encoder, para realizar el transform y el inverse_transform, <br>

detalles enontrados: el segundo mejor modelo fue KNeighborsClassifier , con un rendimiento de KNeighborsClassifier, esto es una puntuacion alta, pero como gano el xgboost, no lo escoji, me sorprende lo <br>
que puede llegar a ser el xgboost