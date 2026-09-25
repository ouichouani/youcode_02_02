from notebooks.silver import silver_data 

fe_data = silver_data.copy()

# TotalSF = TotalBsmtSF + 1stFlrSF + 2ndFlrSF
fe_data["TotalSF"] = (
    fe_data["TotalBsmtSF"] 
    + fe_data["1stFlrSF"] 
    + fe_data["2ndFlrSF"]
    )

# TotalBathrooms = FullBath + 0.5 * HalfBath + BsmtFullBath + 0.5 * BsmtHalfBath
fe_data["TotalBathrooms"] = (
    fe_data["FullBath"]
    + 0.5 * fe_data["HalfBath"]
    + fe_data["BsmtFullBath"]
    + 0.5 * fe_data["BsmtHalfBath"]
)

# HouseAge = YrSold - YearBuilt
fe_data["HouseAge"] = (
    fe_data["YrSold"] - fe_data["YearBuilt"]
)

# TotalPorchSF = OpenPorchSF + EnclosedPorch + 3SsnPorch + ScreenPorch + WoodDeckSF
fe_data["TotalPorchSF"] = (
    fe_data["OpenPorchSF"]
    + fe_data["EnclosedPorch"]
    + fe_data["3SsnPorch"]
    + fe_data["ScreenPorch"]
    + fe_data["WoodDeckSF"]
)


def report():

    print("feature engineering data report".center( 80 , '='))
    print('\n')

    print("feature engineering data shape".center( 80 , '-'))
    print(fe_data.shape)

    print("feature engineering data columns".center( 80 , '-'))
    print(fe_data.columns.to_list())

    print("feature engineering data info".center( 80 , '-'))
    fe_data.info()

    print("feature engineering data describe".center( 80 , '-'))
    print(fe_data.describe())

    print("feature engineering data null values".center( 80 , '-'))
    print(fe_data.isnull().sum())

    print("feature engineering data duplicate values".center( 80 , '-'))
    print(fe_data.duplicated().sum())

    # checking the distribution of SalePrice

    print("total surface statistics".center(80, '-'))
    print(fe_data["TotalSF"].describe())

    print("total surface missing values".center(80, '-'))
    print(fe_data["TotalSF"].isnull().sum())

    print("total sf statistics".center(80, '-'))
    print(fe_data["TotalBathrooms"].describe())

    print("total bathrooms missing values".center(80, '-'))
    print(fe_data["TotalBathrooms"].isnull().sum())

    print("house age statistics".center(80, '-'))
    print(fe_data["HouseAge"].describe())

    print("house age missing values".center(80, '-'))
    print(fe_data["HouseAge"].isnull().sum())

    print("total porch statistics".center(80, '-'))
    print(fe_data["TotalPorchSF"].describe())

    print("total porch missing values".center(80, '-'))
    print(fe_data["TotalPorchSF"].isnull().sum())

if __name__ == "__main__" :
    report()