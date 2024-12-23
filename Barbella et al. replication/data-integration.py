import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.sparse import csr_matrix
import numpy as np
from minisom import MiniSom
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Levenshtein import distance as levenshtein_distance

import nltk
nltk.download('stopwords')

class DataIntegration:
    def __init__(self):
        self.ps = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.tdm = None
        self.tables = {}
        
    def tables_characteristics(self, tables_dict):
        """Get the shapes of the tables"""
        shapes_tables = {}
        for name, table in tables_dict.items():
            shapes_tables[name] = table.shape
        return shapes_tables
    
    def preprocess_text(self, text):
        """Preprocess text by removing special characters, converting to lowercase, etc."""
        # Remove special characters and convert to lowercase
        text = re.sub(r'[^\w\s]', '', str(text).lower())
        
        # Remove stopwords and apply stemming
        words = text.split()
        words = [self.ps.stem(word) for word in words if word not in self.stop_words]
        
        return ' '.join(words)
    
    def load_table(self, name, df):
        """Load a table and preprocess its column names"""
        # Remove columns with null values (FOR REPLICATION)
        # We can set out own threshold value of allowing upto certain %. Uncomment the below code for that.
        # df = df.dropna(axis=1, thresh=threshold)
        df = df.dropna(axis=1)
        
        # Store original table
        self.tables[name] = df
        
        # Preprocess column names
        processed_cols = ' '.join([self.preprocess_text(col) for col in df.columns])
        return processed_cols
    
    def create_tdm(self, documents):
        """Create Term-Document Matrix using TF-IDF"""
        vectorizer = TfidfVectorizer()
        self.tdm = vectorizer.fit_transform(documents)
        return self.tdm
    
    def calculate_levenshtein_matrix(self, documents):
        """Calculate Levenshtein Distance Matrix"""
        n = len(documents)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(n):
                dist_matrix[i][j] = levenshtein_distance(documents[i], documents[j])
                
        return dist_matrix
    
    def perform_hierarchical_clustering(self, distance_matrix):
        """Perform hierarchical clustering"""
        linkage_matrix = linkage(squareform(distance_matrix), method='complete')
        return linkage_matrix
    
    def visualize_heirarchical_clustering(self, linkage_matrix, syn = "syn"):
        """Create dendrograms for heirarchical clustering"""
        # Create the dendrogram
        plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        
        # Customize the plot
        plt.title('Hierarchical Clustering Dendrogram')
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        
        # Show the plot
        plt.savefig(f"{syn}_dendrogram.png")
    
    def perform_kmeans_clustering(self, documents, matrix, n_clusters):
        """Perform k-means clustering with constraints"""
        best_bss_tss = 0
        best_labels = None
        
        for _ in range(5):  # Minimum 5 iterations
            kmeans = KMeans(n_clusters=n_clusters, n_init=10)
            labels = kmeans.fit_predict(matrix)
            
            # Calculate BSS/TSS
            tss = np.sum(np.sum(np.square(matrix - np.mean(matrix, axis=0)), axis=1))
            bss = kmeans.inertia_
            bss_tss = bss / tss if tss != 0 else 0
            
            if bss_tss > best_bss_tss:
                best_bss_tss = bss_tss
                best_labels = labels

            # Check for clusters with only 1 document
            unique_labels, counts = np.unique(labels, return_counts=True)
            single_doc_clusters = unique_labels[counts == 1]
            
            if len(single_doc_clusters) > 0:
                print(f"\nNNNNNNNEEEEEEHHHHHHHH:- {single_doc_clusters}")
                for single_cluster in single_doc_clusters:
                    # Get the indices of documents in this cluster
                    removed_docs = np.where(labels == single_cluster)[0]
                    print(f"Removing document(s) in cluster {single_cluster}: {[documents[i] for i in removed_docs]}")
                # Remove documents in single-document clusters
                mask = np.isin(labels, single_doc_clusters, invert=True)
                matrix = matrix[mask]
                labels = labels[mask]
                
                # Remap remaining labels to be consecutive
                unique_remaining_labels = np.unique(labels)
                label_map = {old: new for new, old in enumerate(unique_remaining_labels)}
                labels = np.array([label_map[l] for l in labels])
                
                # Update n_clusters
                n_clusters -= len(single_doc_clusters)
                
                if n_clusters < 1:
                    # If we're down to 1 or fewer clusters, stop the process
                    break
        return best_labels, best_bss_tss
    
    def train_som(self, tdm, map_size, sparsity_ratio = 0.95):
        """Train Self-Organizing Map"""
        # Removing sparse terms with threshold of .95
        term_sums = np.array(tdm.sum(axis = 0)).ravel()
        mask = term_sums < sparsity_ratio
        new_tdm = tdm[:,mask]
        
        # Training the SOM
        som = MiniSom(map_size[0], map_size[1], new_tdm.shape[1], sigma=1.0, learning_rate=0.5)
        som.random_weights_init(new_tdm)
        som.train_random(new_tdm, 100)
        return som
    
    def calculate_covariance_matrix(self, som_weights, size):
        """Calculate covariance matrix from SOM weights"""
        # Reshaping the weights matrix
        new_som_weights = som_weights.reshape(-1, size)
        # Normalize weights to [0,1]
        normalized_weights = (new_som_weights - np.min(new_som_weights)) / (np.max(new_som_weights) - np.min(new_som_weights))
        
        # Set non-significant elements to 0
        threshold = np.max(normalized_weights) * 0.1
        normalized_weights[normalized_weights < threshold] = 0
        
        return np.cov(normalized_weights.T)
    
    def calculate_integration_index(self, syn_bss_tss, sem_bss_tss, alpha=0.4, beta=0.6):
        """Calculate integration index (ISynSem)"""
        return alpha * syn_bss_tss + beta * sem_bss_tss
            

    def find_matching_columns(self, table1, table2):
        """Find matching columns between two tables"""
        matches = {}
        
        for col1 in table1.columns:
            for col2 in table2.columns:
                # Compare non-numeric columns
                if table1[col1].dtype == object and table2[col2].dtype == object:
                    # Count matching values
                    common_values = len(set(table1[col1].dropna()) & set(table2[col2].dropna()))
                    if common_values > 0:
                        matches[(col1, col2)] = common_values
                        
        return matches
    
    def integrate_tables(self, table1, table2, matching_columns):
        """Integrate two tables based on matching columns"""
        # Sort matching columns by number of common values
        sorted_matches = sorted(matching_columns.items(), key=lambda x: x[1], reverse=True)
        
        if not sorted_matches:
            return None
            
        # Use the best matching columns as join keys
        key1, key2 = sorted_matches[0][0]
        
        # Perform full outer join
        merged = pd.merge(table1, table2, 
                         left_on=key1, 
                         right_on=key2, 
                         how='inner',
                         suffixes=('_1', '_2'))
                         
        return merged
    
    def process_integration(self, tables_dict):
        # Step 0: Get the initial distribution of the tables
        shapes = self.tables_characteristics(tables_dict)
        print(f"\nSTEP 0:- Shapes of the tables:\n{shapes}")

        """Main integration process"""
        # Step 1: Preprocess all tables
        documents = []
        for name, df in tables_dict.items():
            processed_text = self.load_table(name, df)
            documents.append(processed_text)
        print(f"\nSTEP 1:- PREPROCESSING DOCS\n{documents}")
        
        # Step 2: Create TDM
        tdm = self.create_tdm(documents)
        print(f"\nSTEP 2:- TDM\n{(csr_matrix(tdm)).toarray()}")
        
        # Step 3: Syntactic Analysis
        lev_matrix = self.calculate_levenshtein_matrix(documents)
        print(f"\nSTEP 3.1:- LEVENSHTEIN DIST.\n{lev_matrix}")
        hierarch_clusters = self.perform_hierarchical_clustering(lev_matrix)
        print(f"\nSTEP 3.2:- DENDROGRAM SYN. DONE and LINKAGE MATRIX\n{hierarch_clusters}")
        self.visualize_heirarchical_clustering(hierarch_clusters)
        # syn_labels, syn_bss_tss = self.perform_kmeans_clustering(documents, lev_matrix, n_clusters=2)
        syn_labels, syn_bss_tss = self.perform_kmeans_clustering(documents, lev_matrix, \
            n_clusters=len(documents)//2 if len(documents) > 3 else 2)
        print(f"\nSTEP 3.3:- KMEANS SYN.\n{syn_labels,syn_bss_tss}")
        
        # Step 4: Semantic Analysis
        som = self.train_som(tdm.toarray(), map_size=(len(documents),len(documents)))
        print(f"\nSTEP 4.1:- SOM WEIGHTS\n{som.get_weights()}")
        print(f"\nSTEP 4.1:- SOM SHAPE\n{som.get_weights().shape}")
        cov_matrix = self.calculate_covariance_matrix(som.get_weights(), size=len(documents))
        self.visualize_heirarchical_clustering(hierarch_clusters, "sem")
        print(f"\nSTEP 4.2:- COV MATRIX\n{cov_matrix}")
        # sem_labels, sem_bss_tss = self.perform_kmeans_clustering(documents, cov_matrix, n_clusters=2)
        sem_labels, sem_bss_tss = self.perform_kmeans_clustering(documents, cov_matrix, \
            n_clusters=len(documents)//2 if len(documents) > 3 else 2)
        print(f"\nSTEP 4.3:- KMEANS SEM.\n{sem_labels,sem_bss_tss}")
        
        # Step 5: Calculate Integration Index
        integration_index = self.calculate_integration_index(syn_bss_tss, sem_bss_tss)
        
        """
        TODO:- Display the clusters and understand what clusters are common, indicating those to be merged
        """
        # Step 6: Find and Integrate Similar Tables
        results = []
        for i in range(len(syn_labels)):
            for j in range(i+1, len(syn_labels)):
                if syn_labels[i] == syn_labels[j] and sem_labels[i] == sem_labels[j]:
                    table1_name = list(tables_dict.keys())[i]
                    table2_name = list(tables_dict.keys())[j]
                    
                    matches = self.find_matching_columns(
                        self.tables[table1_name],
                        self.tables[table2_name]
                    )
                    
                    if matches:
                        integrated_table = self.integrate_tables(
                            self.tables[table1_name],
                            self.tables[table2_name],
                            matches
                        )
                        results.append({
                            'tables': (table1_name, table2_name),
                            'matching_columns': matches,
                            'integrated_table': integrated_table
                        })
        
        print(f"\nMERGES:- {len(results)}")
        return {
            'integration_index': integration_index,
            'syntactic_bss_tss': syn_bss_tss,
            'semantic_bss_tss': sem_bss_tss,
            'integrated_results': results
        }

# Example usage:
if __name__ == "__main__":
    heatdf = pd.read_csv("datasets/Heat_vulnerability/NYC_Heat_vulnerability_index.csv")
    treedf = pd.read_csv("datasets/Tree_Census/new_york_tree_census_2015.csv")
    trafficdf = pd.read_csv("datasets/Traffic_Road/Automated_Traffic_Volume_Counts_20241029.csv")
    ricettedf = pd.read_csv("datasets/Food/ricette.csv")
    prododf = pd.read_csv("datasets/Food/prodottitradizionali.csv")
    
    # Initialize and run integration
    integrator = DataIntegration()
    results = integrator.process_integration({
        'trees': treedf,
        'heat': heatdf,
        'traffic': trafficdf
        # 'ricette': ricettedf,
        # 'prodo': prododf
    })


    
    print(f"Integration Index: {results['integration_index']:.2f}")
    print(f"Syntactic BSS/TSS: {results['syntactic_bss_tss']:.2f}")
    print(f"Semantic BSS/TSS: {results['semantic_bss_tss']:.2f}")
    
    for result in results['integrated_results']:
        print(f"\nIntegrated tables: {result['tables']}")
        print(f"Matching columns: {result['matching_columns']}")
        print(f"\nIntegrated table shape: {result['integrated_table'].shape}")
        # print(result['integrated_table'])
        result['integrated_table'].to_csv("merged_data.csv", index=False)
