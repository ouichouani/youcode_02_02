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

from sklearn.model_selection import cross_val_score
fe_data = fe_data.iloc[:1461]

# Target and features
x = fe_data.drop(columns=["Id", "SalePrice"])
y = fe_data["SalePrice"]

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=100
)

# Identify columns
numerical_features = X_train.select_dtypes(
    include=["int64", "float64"]
).columns

categorical_features = X_train.select_dtypes(
    include=["object" , "str"]
).columns


# Numerical preprocessing
numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical preprocessing
categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

# Apply the correct preprocessing to each type of column
preprocessor = ColumnTransformer([
    ("num", numerical_pipeline, numerical_features),
    ("cat", categorical_pipeline, categorical_features)
])

model_pipline = Pipeline([
    ("preprocessor" , preprocessor ) ,
    ("model" , LinearRegression() )
])

# Learn preprocessing from training data and transform both datasets
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)


def model_LinearRegression() :
    model_LinearRegression = LinearRegression()
    model_LinearRegression.fit(X_train_processed , y_train)
    return model_LinearRegression

def model_RandomTrees() :
    model_RandomForest = RandomForestRegressor( n_estimators=100, random_state=100)
    model_RandomForest.fit(X_train_processed , y_train)
    return model_RandomForest

def model_GradientBoostingRegressor () :
    model_GradientBoostingRegressor = GradientBoostingRegressor(random_state=100)
    model_GradientBoostingRegressor.fit(X_train_processed , y_train)
    return model_GradientBoostingRegressor

def evaluate(model) :
    y_pred = model.predict(X_test_processed)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae , rmse , r2

def evaluate_cv(model_object) :
    result_obj = {}
    for name , model in model_object.items() :
        score = cross_val_score(model_pipline, X_train , y_train , cv=5 , scoring="neg_mean_absolute_error"  ) 
        # score = cross_val_score(model , X_train_processed , y_train , cv=5 , scoring="neg_mean_absolute_error"  ) 
        result_obj[name] = -score.mean()
    return result_obj

def report(models):
    for model in models :
        mae , rmse , r2 = evaluate(model)
        print("result of the matrics of the tree models")
        print(' - linear regression')
        print(' - mae :', mae)
        print(' - rmse :', rmse)
        print(' - r2 :', r2)

model_oblect = {
    'model_LinearRegression' : LinearRegression() ,
    'model_RandomForest' : RandomForestRegressor(n_estimators=100 , random_state=100) ,
    'model_GradientBoostingRegressor' : GradientBoostingRegressor(random_state=100) ,
}

# print(evaluate_cv(model_oblect))