import streamlit as st
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import numpy as np
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure the 'Files' directory exists
if not os.path.exists("Files"):
    os.makedirs("Files")

class DataAnalysis:
    def _init_(self):
        self.df = None
        self.base_path = 'Files/'  # Default base path for files

    def load_data(self, file_path=None):
        """Load dataset from file or Streamlit uploader."""
        if file_path is None:
            file_path = os.path.join(self.base_path, 'dataset.csv')  # Adjust the file path here
        
        try:
            self.df = pd.read_csv(file_path)
            st.success(f"Dataset loaded successfully from {file_path}. Shape: {self.df.shape}")
            st.dataframe(self.df.head())  # Display the loaded data for verification
        except Exception as e:
            st.error(f"Error loading dataset from path: {str(e)}")

    def show_data(self, file_path=None):
        """Show the first few rows of the raw data."""
        if file_path is None:
            file_path = os.path.join(self.base_path, 'dataset.csv')  # Adjust the file path here
        
        try:
            self.df = pd.read_csv(file_path)
            st.write("First 10 rows of the dataset:")
            st.dataframe(self.df.head(10))
        except Exception as e:
            st.error(f"Error loading raw data from {file_path}: {str(e)}")

    def plot_country_distribution(self, file_path=None):
        """Plot the distribution of transactions by country."""
        if file_path is None:
            file_path = os.path.join(self.base_path, 'dataset.csv')  # Adjust the file path here
        
        if self.df is None:
            st.warning("No data loaded. Please load data first.")
            return

        if 'Country' in self.df.columns:
            plt.figure(figsize=(9, 10))
            sns.countplot(y='Country', data=self.df, order=self.df['Country'].value_counts().index)
            plt.title("Number of Transactions by Country")
            st.pyplot(plt)
        else:
            st.error("Column 'Country' not found in the dataset.")

    def clean_data(self, file_path=None):
        """Clean the dataset by handling missing values and generating 'Sales'."""
        if file_path is None:
            file_path = os.path.join(self.base_path, 'dataset.csv')  # Adjust the file path here
        
        if self.df is not None:
            try:
                self.df = self.df.dropna(subset=['CustomerID'])  # Drop rows with missing CustomerID
                self.df['CustomerID'] = self.df['CustomerID'].astype(int)
                self.df['Sales'] = self.df['Quantity'] * self.df['UnitPrice']
                self.df.to_csv(os.path.join(self.base_path, 'cleaned_transactions.csv'), index=False)  # Adjust the save path here
                st.success("Data cleaned and saved as 'Files/cleaned_transactions.csv'.")
                st.dataframe(self.df.head())  # Display cleaned data for verification
            except Exception as e:
                st.error(f"Error during data cleaning: {str(e)}")
        else:
            st.warning("No data loaded. Please load data first.")

    def group_by_customer(self, file_path=None):
        """Aggregate data by CustomerID."""
        if file_path is None:
            file_path = os.path.join(self.base_path, 'cleaned_transactions.csv')  # Adjust the file path here
        
        try:
            self.df = pd.read_csv(file_path)
            invoice_data = self.df.groupby('CustomerID').agg(total_transactions=('InvoiceNo', 'nunique'))
            product_data = self.df.groupby('CustomerID').agg(
                total_products=('StockCode', 'count'),
                total_unique_products=('StockCode', 'nunique')
            )
            sales_data = self.df.groupby('CustomerID').agg(
                total_sales=('Sales', 'sum'),
                avg_product_value=('Sales', 'mean')
            )
            invoice_data.to_csv(os.path.join(self.base_path, 'analytical_base_table.csv'))  # Adjust the save path here
            st.success("Aggregated data saved as 'Files/analytical_base_table.csv'.")
            return invoice_data, product_data, sales_data
        except Exception as e:
            st.error(f"Error during grouping: {str(e)}")
            return None, None, None

    def perform_pca(self, file_path=None):
        """Perform PCA on the item data."""
        if file_path is None:
            file_path = os.path.join(self.base_path, 'cleaned_transactions.csv')  # Adjust the file path here
        
        if self.df is not None:
            try:
                item_dummies = pd.get_dummies(self.df['StockCode'])
                item_dummies['CustomerID'] = self.df['CustomerID']
                item_data = item_dummies.groupby('CustomerID').sum()

                top_items = item_data.sum().sort_values(ascending=False).head(120).index
                top_item_data = item_data[top_items]
                top_item_data.to_csv(os.path.join(self.base_path, 'threshold_item_data.csv'))  # Adjust the save path here

                scaler = StandardScaler()
                item_data_scaled = scaler.fit_transform(top_item_data)

                pca = PCA()
                pca.fit(item_data_scaled)

                cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
                plt.figure(figsize=(10, 6))
                plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
                plt.title('Cumulative Explained Variance by Principal Components')
                plt.xlabel('Number of Components')
                plt.ylabel('Cumulative Explained Variance')
                st.pyplot(plt)

                pca_transformed = pca.transform(item_data_scaled)
                pca_df = pd.DataFrame(pca_transformed, columns=[f'PC{i + 1}' for i in range(pca_transformed.shape[1])])
                pca_df.index = top_item_data.index
                pca_df.to_csv(os.path.join(self.base_path, 'pca_item_data.csv'))  # Adjust the save path here

                st.success("PCA completed. Data saved as 'Files/pca_item_data.csv'.")
            except Exception as e:
                st.error(f"Error during PCA: {str(e)}")
        else:
            st.warning("No data loaded. Please load data first.")

    def perform_cluster_analysis(self, file_path=None):
        """Perform KMeans clustering."""
        try:
            if file_path is None:
                file_path = os.path.join(self.base_path, 'cleaned_transactions.csv')  # Adjust the file path here

            # Load the main dataset and the PCA/threshold data
            base_df = pd.read_csv(file_path)
            threshold_data = pd.read_csv(os.path.join(self.base_path, 'threshold_item_data.csv'), index_col=0)  # Adjust the file path here
            pca_data = pd.read_csv(os.path.join(self.base_path, 'pca_item_data.csv'), index_col=0)  # Adjust the file path here

            # Ensure the indices are aligned by matching the 'CustomerID' (or similar identifier)
            threshold_data = threshold_data.loc[base_df['CustomerID'].isin(threshold_data.index)]
            pca_data = pca_data.loc[base_df['CustomerID'].isin(pca_data.index)]

            # Perform scaling on the threshold data
            threshold_scaled = StandardScaler().fit_transform(threshold_data)
            pca_scaled = StandardScaler().fit_transform(pca_data)

            # Fit the KMeans model on the scaled data
            kmeans_threshold = KMeans(n_clusters=3, random_state=123).fit(threshold_scaled)
            kmeans_pca = KMeans(n_clusters=3, random_state=123).fit(pca_scaled)

            # Initialize new columns in base_df for the cluster labels
            base_df['threshold_cluster'] = np.nan
            base_df['pca_cluster'] = np.nan

            # Assign the cluster labels to the correct rows based on 'CustomerID'
            base_df.loc[base_df['CustomerID'].isin(threshold_data.index), 'threshold_cluster'] = kmeans_threshold.labels_
            base_df.loc[base_df['CustomerID'].isin(pca_data.index), 'pca_cluster'] = kmeans_pca.labels_

            # Check if the assignment is successful
            st.success("Clustering completed.")
            st.write("Sample clustered data:", base_df.head())

        except Exception as e:
            st.error(f"Error during clustering: {str(e)}")


def main():
    st.title("Client Clustering Dashboard")

    analyzer = DataAnalysis()

    menu = ["Load Data", "View Data", "Plot Country Distribution", "Clean Data",
            "Group Data by Customer", "Perform PCA", "Perform Cluster Analysis"]
    choice = st.sidebar.selectbox("Select an action", menu)

    # Update this path with the actual location of the dataset
    file_path = "D:\\5thSemester\\DataScience\\Project\\Customer-Segmentation-main (1)\\Customer-Segmentation-main\\Files\\dataset.csv"  # Adjust path here

    if choice == "Load Data":
        analyzer.load_data(file_path=file_path)

    elif choice == "View Data":
        analyzer.show_data(file_path=os.path.join(analyzer.base_path, 'dataset.csv'))

    elif choice == "Plot Country Distribution":
        analyzer.plot_country_distribution(file_path=os.path.join(analyzer.base_path, 'dataset.csv'))

    elif choice == "Clean Data":
        analyzer.clean_data(file_path=os.path.join(analyzer.base_path, 'cleaned_transactions.csv'))  # Adjust path here

    elif choice == "Group Data by Customer":
        invoice_data, product_data, sales_data = analyzer.group_by_customer(file_path=os.path.join(analyzer.base_path, 'cleaned_transactions.csv'))  # Adjust path here
        if invoice_data is not None:
            st.write("Invoice Data", invoice_data.head())
            st.write("Product Data", product_data.head())
            st.write("Sales Data", sales_data.head())

    elif choice == "Perform PCA":
        analyzer.perform_pca(file_path=os.path.join(analyzer.base_path, 'cleaned_transactions.csv'))  # Adjust path here

    elif choice == "Perform Cluster Analysis":
        analyzer.perform_cluster_analysis(file_path=os.path.join(analyzer.base_path, 'cleaned_transactions.csv'))  # Adjust path here


if __name__ == "__main__":
    main()