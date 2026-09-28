from notebooks.feature_engineering import fe_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression

from sklearn.metrics import mean_absolute_error , r2_score , root_mean_squared_error
from sklearn.ensemble  import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
import joblib


fe_data = fe_data.iloc[:1461]
x = fe_data.drop(columns=["Id", "SalePrice"])
y = fe_data["SalePrice"]
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100)

numerical_features = X_train.select_dtypes(include=["int64", "float64"]).columns
categorical_features = X_train.select_dtypes(include=["object" , "str"]).columns

# PREPROCCESSOR

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

# MODELS PIPLINES

model_pipline_LinearRegression = Pipeline([
    ("preprocessor" , preprocessor ) ,
    ("model" , LinearRegression() )
])

model_pipline_RandomForestRegressor = Pipeline([
    ("preprocessor" , preprocessor ) ,
    ("model" , RandomForestRegressor(random_state=100) )
])

model_pipline_GradientBoostingRegressor = Pipeline([
    ("preprocessor" , preprocessor ) ,
    ("model" , GradientBoostingRegressor(random_state=200 , n_estimators=100 ))
])

# SHOUSING THE MODEL

model_object = {
    'model_LinearRegression' : model_pipline_LinearRegression ,
    'model_RandomForest' : model_pipline_RandomForestRegressor ,
    'model_GradientBoostingRegressor' : model_pipline_GradientBoostingRegressor
}

def evaluate(model) :
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae , rmse , r2

def cross_validation(model_object) :
    result_obj = {}
    scoring = {
        "mae" : "neg_mean_absolute_error",
        "rmse" : "neg_root_mean_squared_error" ,
        "r2" : "r2"
    }
    for name , model in model_object.items() :
        score = cross_validate(model, X_train , y_train , cv=5 , scoring=scoring  ) 
        result_obj[name] = {
            "mae" : -score["test_mae"].mean() ,
            "rmse" : -score["test_rmse"].mean() ,
            "r2" : score["test_r2"].mean()
        }
    return result_obj

def report(model_object):
    print('\n')
    print("result of the matrics of the tree models".center(80 , "="))
    for name , model in model_object.items() :
        model.fit(X_train, y_train)
        mae , rmse , r2 = evaluate(model)
        print('- ' , name)
        print(' --  mae :', mae)
        print(' --  rmse :', rmse)
        print(' --  r2 :', r2)
    print('\n')
    print("result of cross validation of the tree models".center(80 , "="))
    result = cross_validation(model_object)
    for name , score in result.items() :
        print('- ' , name)
        print(' --  mae :', score['mae'])
        print(' --  rmse :', score['rmse'])
        print(' --  r2 :', score['r2'])

# BEST PARAMS FOR GRADIANTBOOSTINGREGRESSOR

test_params = {
    "model__n_estimators": [50, 100, 150, 200],
    "model__learning_rate": [0.05, 0.1, 0.15],
    "model__max_depth": [2, 3, 4]
}
grid_search_cv = GridSearchCV(model_pipline_GradientBoostingRegressor , param_grid=test_params , cv=5  , scoring="neg_mean_absolute_error")
grid_search_cv.fit(X_train , y_train)
best_model = grid_search_cv.best_estimator_

# CREATE AN ORIGINAL GRADIANTBOOSTINGREGRESSOR MODEL TO TEST
# original_model = model_pipline_GradientBoostingRegressor
# original_model.fit(X_train , y_train)

# COMPARE WITH ORIGINAL GRADIANTBOOSTINGREGRESSOR MODEL
# y_pred_tuned = best_model.predict(X_test)
# y_pred_original = original_model.predict(X_test)

# mae_tuned = mean_absolute_error(y_test , y_pred_tuned )
# rmse_tuned = root_mean_squared_error(y_test , y_pred_tuned )
# r2_tuned = r2_score(y_test , y_pred_tuned )

# mae_original = mean_absolute_error(y_test , y_pred_original )
# rmse_original = root_mean_squared_error(y_test , y_pred_original )
# r2_original = r2_score(y_test , y_pred_original )



# print("Original GradientBoostingRegressor")
# print("MAE :", mae_original)
# print("RMSE:", rmse_original)
# print("R²  :", r2_original)

# print("\nTuned GradientBoostingRegressor")
# print("MAE :", mae_tuned)
# print("RMSE:", rmse_tuned)
# print("R²  :", r2_tuned)

# print("\nBest parameters:")
# print(grid_search_cv.best_params_)


joblib.dump(best_model, "models/house_price_model.joblib")
