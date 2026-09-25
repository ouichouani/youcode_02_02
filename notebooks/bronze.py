import pandas as pd

bronze_data = pd.read_csv('data/House_Prices.csv')

def report():

    print("bronze data report".center( 80 , '='))
    print('\n')

    print("bronze data shape".center( 80 , '-'))
    print(bronze_data.shape)

    print("bronze data columns".center( 80 , '-'))
    print(bronze_data.columns.to_list())

    print("bronze data info".center( 80 , '-'))
    bronze_data.info()

    print("bronze data describe".center( 80 , '-'))
    print(bronze_data.describe())

    print("bronze data null values".center( 80 , '-'))
    print(bronze_data.isnull().sum())

    print("bronze data duplicate values".center( 80 , '-'))
    print(bronze_data.duplicated().sum())

    # checking the distribution of SalePrice

    print("sale price statistics".center(80, '-'))
    print(bronze_data["SalePrice"].describe())

    print("sale price missing values".center(80, '-'))
    print(bronze_data["SalePrice"].isnull().sum())

    print("id unique values".center(80, '-'))
    print(bronze_data["Id"].nunique())

    print("data types".center(80, '-'))
    print(bronze_data.dtypes.value_counts())

    print("missing values".center(80, '-'))
    missing = bronze_data.isnull().sum()
    print(missing[missing > 0].sort_values(ascending=False))

if __name__ == "__main__":
    report()
