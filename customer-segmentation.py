import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 20)  
pd.set_option('display.float_format', '{:.2f}'.format)  

from matplotlib import pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


df = pd.read_csv('Files/dataset.csv')
print("Dataset Shape:", df.shape)
print("First 10 rows of the dataset:\n", df.head(10))


plt.figure(figsize=(9, 10))
sns.countplot(y='Country', data=df)
plt.title("Number of Transactions by Country")
plt.show()


print("\nMissing Values:\n", df.isnull().sum())


df = df[df.CustomerID.notnull()]
df['CustomerID'] = df.CustomerID.astype(int)
print("\nFirst 5 Customer IDs:\n", df.CustomerID.head())


df['Sales'] = df.Quantity * df.UnitPrice
print("\nFirst 5 Sales Values:\n", df.Sales.head())


df.to_csv('Files/cleaned_transactions.csv', index=None)


invoice_data = df.groupby('CustomerID').agg(total_transactions=('InvoiceNo', 'nunique'))
print("\nInvoice Data:\n", invoice_data.head())


product_data = df.groupby('CustomerID').agg(
    total_products=('StockCode', 'count'),
    total_unique_products=('StockCode', 'nunique')
)
print("\nProduct Data:\n", product_data.head())


sales_data = df.groupby('CustomerID').agg(
    total_sales=('Sales', 'sum'),
    avg_product_value=('Sales', 'mean')
)
print("\nSales Data:\n", sales_data.head())


plt.figure(figsize=(9, 6))
sns.barplot(x=invoice_data.index[:10], y=invoice_data['total_transactions'][:10])
plt.title("Top 10 Customers by Total Transactions")
plt.xlabel("CustomerID")
plt.ylabel("Total Transactions")
plt.xticks(rotation=45)
plt.show()

cart_data = df.groupby(['CustomerID', 'InvoiceNo']).agg(cart_value=('Sales', 'sum'))


print(cart_data.head(20))


cart_data.reset_index(inplace=True)


print(cart_data.head(10))


agg_cart_data = cart_data.groupby('CustomerID').agg(
    avg_cart_value=('cart_value', 'mean'),
    min_cart_value=('cart_value', 'min'),
    max_cart_value=('cart_value', 'max')
)


print(agg_cart_data.head())


customer_df = invoice_data.join([product_data, sales_data, agg_cart_data])


print(customer_df.head())


customer_df.to_csv('Files/analytical_base_table.csv')



df = pd.read_csv('Files/cleaned_transactions.csv')


item_dummies = pd.get_dummies(df.StockCode)
item_dummies['CustomerID'] = df.CustomerID


item_data = item_dummies.groupby('CustomerID').sum()

print("Item data sample (first 5 rows):")
print(item_data.head())


top_120_items = item_data.sum().sort_values().tail(120).index
print("\nTop 120 items by purchase frequency:")
print(top_120_items)


top_120_item_data = item_data[top_120_items]
print("\nFiltered data (top 120 items, first 5 rows):")
print(top_120_item_data.head())


top_120_item_data.to_csv('Files/threshold_item_data.csv')


item_data = pd.read_csv('Files/threshold_item_data.csv', index_col=0)
print("\nItem data shape after reloading:", item_data.shape)


scaler = StandardScaler()
item_data_scaled = scaler.fit_transform(item_data)

print("\nScaled item data (first 5 rows):")
print(item_data_scaled[:5])


pca = PCA()
pca.fit(item_data_scaled)


PC_items = pca.transform(item_data_scaled)

print("\nPrincipal Components (first 5 rows):")
print(PC_items[:5])


cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)


plt.figure(figsize=(10, 6))
plt.grid()
plt.plot(range(len(cumulative_explained_variance)), cumulative_explained_variance, marker='o')
plt.title('Cumulative Explained Variance by Principal Components')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

print("\nCumulative explained variance for 300 components:")
if len(cumulative_explained_variance) > 300:
    print(cumulative_explained_variance[300])
else:
    print("Less than 300 components available.")


n_components = min(300, item_data_scaled.shape[1])  
pca = PCA(n_components=n_components)
PC_items = pca.fit_transform(item_data_scaled)

print("\nReduced principal components shape:", PC_items.shape)


items_pca = pd.DataFrame(PC_items)
items_pca.columns = ['PC{}'.format(i + 1) for i in range(PC_items.shape[1])]
items_pca.index = item_data.index

print("\nPCA-transformed data (first 5 rows):")
print(items_pca.head())


items_pca.to_csv('Files/pca_item_data.csv')
print("\nPCA-transformed data saved to 'Files/pca_item_data.csv'")

base_df = pd.read_csv('Files/analytical_base_table.csv', index_col=0)


threshold_item_data = pd.read_csv('Files/threshold_item_data.csv', index_col=0)


pca_item_data = pd.read_csv('Files/pca_item_data.csv', index_col=0)

print( base_df.shape )
print( threshold_item_data.shape )
print( pca_item_data.shape )

threshold_df = base_df.join(threshold_item_data)

threshold_df.head()

pca_df = base_df.join(pca_item_data)


pca_df.head()

t_scaler = StandardScaler()
p_scaler = StandardScaler()


threshold_df_scaled = t_scaler.fit_transform(threshold_df)
pca_df_scaled = p_scaler.fit_transform(pca_df)
t_kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 123)
t_kmeans.fit(threshold_df_scaled)
threshold_df['cluster'] = t_kmeans.fit_predict(threshold_df_scaled)

sns.lmplot(x='total_sales', y='avg_cart_value', hue='cluster', data=threshold_df, fit_reg=False)

plt.title('Cluster Analysis: Total Sales vs Average Cart Value')
plt.xlabel('Total Sales')
plt.ylabel('Average Cart Value')


plt.show()
p_kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 123)
p_kmeans.fit(pca_df_scaled)
pca_df['cluster'] = p_kmeans.fit_predict(pca_df_scaled)

sns.lmplot(x='total_sales', y='avg_cart_value', hue='cluster', data=pca_df, fit_reg=False)

adjusted_rand_score(pca_df.cluster, threshold_df.cluster)