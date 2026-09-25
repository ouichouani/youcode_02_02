
from notebooks.bronze import bronze_data

silver_data = bronze_data.copy()

numerical_features =[]
categorical_features =[]
target = "SalePrice"
id_column="Id"

categorical_as_numeric = [
    "MSSubClass",
    "MoSold"
]

# classify numerical and categorical features
def classify_features():

    for column in silver_data.columns :
        if column == target or column == id_column:
            continue

        if str(silver_data[column].dtype) in ["int64", "float64"] : 
            numerical_features.append(column)
        else : 
            categorical_features.append(column)

    for column in categorical_as_numeric:
        numerical_features.remove(column)
        categorical_features.append(column)

# fill nan values
def normalize_data():
    # fill nan value in numerical data
    for col in numerical_features :
        silver_data[col] = silver_data[col].fillna(
            silver_data[col].median()
        )

    # fill nan value in string data
    for col in categorical_features :
        silver_data[col] = silver_data[col].fillna(
            silver_data[col].mode()[0]
        )


classify_features()
normalize_data()


