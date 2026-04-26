import pandas as pd
import matplotlib.pyplot as plt


newdf = pd.read_csv(".csv")

print("First 5 rows:\n", newdf.head())
print("\nInfo:")
print(newdf.info())


newdf.ffill(inplace=True)
newdf.drop_duplicates(inplace=True)


num_cols = newdf.select_dtypes(include='number').columns
cat_cols = newdf.select_dtypes(include='object').columns

print("\nNumeric Columns:", num_cols)
print("Categorical Columns:", cat_cols)


if len(num_cols) > 0:
    print("\nNUMERIC STATS")
    print("Mean:\n", newdf[num_cols].mean())
    print("Median:\n", newdf[num_cols].median())
    print("Mode:\n", newdf[num_cols].mode().iloc[0])

    print("Variance:\n", newdf[num_cols].var())
    print("Standard Deviation:\n", newdf[num_cols].std())

    
    newnum = newdf[num_cols].iloc[:, :5]

    
    newnum.hist(figsize=(6,6))
    plt.suptitle("Histogram")
    plt.show()

    
    plt.figure(figsize=(6,6))
    newnum.boxplot()
    plt.title("Box Plot")
    plt.xticks(rotation=45)
    plt.show()

   
    corr = newnum.corr()
    plt.figure(figsize=(6,6))
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Heatmap")
    plt.show()

    
    if newnum.shape[1] >= 2:
        plt.figure(figsize=(6,6))
        plt.scatter(newnum.iloc[:,0], newnum.iloc[:,1])
        plt.title("Scatter Plot")
        plt.show()


if len(cat_cols) > 0:
    print("\nCATEGORICAL STATS")
    
    for col in cat_cols[:2]:  
        print(f"\nColumn: {col}")
        print(newdf[col].value_counts())

        
        plt.figure(figsize=(6,6))
        newdf[col].value_counts().plot(kind='bar')
        plt.title(f"Bar Chart - {col}")
        plt.xticks(rotation=90)
        plt.show()

        
        counts = newdf[col].value_counts()


        top_counts = counts[:5]
        others = counts[5:].sum()

        top_counts["Others"] = others

        plt.figure(figsize=(6,6))
        top_counts.plot(kind='pie', autopct='%1.1f%%')
        plt.title(f"Pie Chart - {col}")
        plt.ylabel("")
        plt.show()