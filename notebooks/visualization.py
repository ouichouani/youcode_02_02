from notebooks.silver import silver_data , numerical_features , target 
import matplotlib.pyplot as plt
import seaborn as sns


# SalePrice
def diagramme_saleprice():
    plt.figure(figsize=(10, 6))
    sns.histplot(silver_data[target], kde=True)
    plt.title("SalePrice Distribution")
    plt.xlabel("Sale Price")
    plt.ylabel("Frequency")
    plt.show()

# GrLivArea vs SalePrice
def diagramme_grlivarea():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=silver_data,
        x="GrLivArea",
        y=target
    )
    plt.title("Living Area vs Sale Price")
    plt.xlabel("Living Area")
    plt.ylabel("Sale Price")
    plt.show()

# qualité vs prix
def diagramme_qualite():
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=silver_data,
        x="OverallQual",
        y=target
    )
    plt.title("Overall Quality vs Sale Price")
    plt.xlabel("Overall Quality")
    plt.ylabel("Sale Price")
    plt.show()

# prix selon les quartiers 
def diagramme_quartiers():
    plt.figure(figsize=(10, 6))
    sns.boxplot(
        data=silver_data,
        x="Neighborhood",
        y=target
    )
    plt.title("Neighborhood vs Sale Price")
    plt.xlabel("Neighborhood")
    plt.ylabel("Sale Price")
    plt.xticks(rotation=45)
    plt.show()

# année de construction vs prix ;
def diagramme_annee_construction():
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=silver_data,
        x="YearBuilt",
        y=target
    )
    plt.title("Built Year vs Sale Price")
    plt.xlabel("année de construction")
    plt.ylabel("Sale Price")
    plt.show()

# matrice de corrélation.
def correlation_matrix():
    correlation = silver_data[numerical_features + [target]].corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        correlation[[target]],
        cmap="coolwarm",
        fmt=".2f",
        annot=True,
    )
    plt.title("Correlation Matrix")
    plt.show()

# get each row and it's outrange value counts
def out_of_range_columns():
    print("\nColumns with outliers:")
    columns = numerical_features + [target]
    for column in columns :
        Q1 = silver_data[column].quantile(0.25)
        Q3 = silver_data[column].quantile(0.75)

        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = ((silver_data[column] < lower) | (silver_data[column] > upper)).sum()

        if outliers > 0:
            print(f" - {column}: {outliers} outliers")



if __name__ == "__main__":

    diagramme_saleprice()
    diagramme_grlivarea()
    diagramme_qualite()
    diagramme_quartiers()
    diagramme_annee_construction()
    correlation_matrix()

    out_of_range_columns()