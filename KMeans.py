import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt 
from random import choices


np.random.seed(100)
df = pd.read_csv('A2Q1.csv',header = None)
data_matrix = np.array(df)


class KMeansClustering:
    def __init__(self, k):
        self.K = k
        self.plot_figure = True
        self.num_data = data_matrix.shape[0]
        self.num_features = data_matrix.shape[1]
        self.error = []
        self.count = 0
    
    

    def make_clusters(self, centroids):
        # Will contain a list of the points that are associated with that specific cluster
        clusters = [[] for _ in range(self.K)]

        # Loop through each point and check which is the closest cluster
        for dp_id, dp in enumerate(data_matrix):
            nearest_centroid = np.argmin(np.sqrt(np.sum((dp - centroids) ** 2, axis=1)))
            clusters[nearest_centroid].append(dp_id)

        return clusters
    
    def init_rand_centroids(self):
        centroids = np.zeros((self.K, self.num_features))
        centroids = choices(data_matrix,k=4)
        return centroids

    def calculate_new_centroids(self, clusters):
        centroids = np.zeros((self.K, self.num_features))
        for i, c in enumerate(clusters):
            new_centroid = np.mean(data_matrix[c], axis=0)
            centroids[i] = new_centroid

        return centroids

    def new_cluster(self, clusters):
        temp = np.zeros(self.num_data)

        for c_id, c in enumerate(clusters):
            for i in c:
                temp[i] = c_id

        return temp
    
        
    def fit(self):
        centroids = self.init_rand_centroids()

        for it in range(100):
            clusters = self.make_clusters(centroids)

            previous_centroids = centroids
            
            dist = np.zeros(data_matrix.shape[0])
            for idx, sample in enumerate(data_matrix):
                cetroidID = np.argmin(np.sqrt(np.sum((sample - centroids) ** 2, axis=1)))
                dist[idx] = np.linalg.norm(sample - centroids[cetroidID])  
                
            self.error.append(np.sum(dist))
            self.count = self.count+1
            
            
            centroids = self.calculate_new_centroids(clusters)

            change = centroids - previous_centroids

            if not change.any():
                break
                
          
        yAxis = np.arange(self.count-1)
        xAxis = np.arange(self.count-1)
        
        for i in range(self.count-1):
            xAxis[i] = self.error[i]
        
        plt.scatter(yAxis,xAxis)
        plt.plot(yAxis,xAxis)
        plt.title('Error Function')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()
             

        # Get label predictions
        y_pred = self.new_cluster(clusters)
        return y_pred
        
        

if __name__ == "__main__":
    
    k = 4
    Kmeans = KMeansClustering(k)
    predict = Kmeans.fit()
        
