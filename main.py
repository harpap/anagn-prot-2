import sys
import argparse
import data_parser
import numpy as np
from sklearn import metrics
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering

def main(args):

    def purity_score(y_true, y_pred):
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        print('sum')
        print(contingency_matrix)
        print('amax')
        print(np.amax(contingency_matrix, axis=0))
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

    def f1_score(y_true, y_pred, max_clusters):
        y_true_c = []
        y_pred_c = []
        sum = 0
        for i in range(max_clusters):
            for j in range(len(y_true)):
                if (i==y_pred[j]):
                    y_pred_c.append(y_pred[j])
                    y_true_c.append(y_true[j])
            counts = np.bincount(y_true_c)
            y_pred_c = [np.argmax(counts)] * len(y_pred_c)
            sum += metrics.f1_score(y_true_c, y_pred_c)
            print('y_pred_c')
            print(y_pred_c)
            print('y_true_c')
            print(y_true_c)
            print('inside f1')
            print (metrics.f1_score(y_true_c, y_pred_c))
            y_true_c = []
            y_pred_c = []
        return sum

    def kMNS():
        for n_clusters in range_n_clusters:
            # Initialize the clusterer with n_clusters value and a random generator
            # seed of 10 for reproducibility.
            # 10 Number of times the k-means algorithm will be run with different centroid seeds.
            # The final results will be the best output of 10 consecutive runs in terms of inertia.
            clusterer = KMeans(n_clusters=n_clusters, random_state=10)
            cluster_labels = clusterer.fit_predict(data)
            #print('purity')
            #print (purity_score(target, cluster_labels))
            print('f1')
            print (f1_score(target, cluster_labels, n_clusters))
    
    def SPC():
        #for n_clusters in range_n_clusters:
            clusterer = SpectralClustering(n_clusters=2, assign_labels="discretize", random_state=0)
            cluster_labels = clusterer.fit_predict(data[:100])
            print(cluster_labels)
    
    def AMC():
        for n_clusters in range_n_clusters:
            clusterer = AgglomerativeClustering(n_clusters=2)
            cluster_labels = clusterer.fit_predict(data[:100])
    
    range_n_clusters = [2, 4, 8]
    options = {'occupancy1' : data_parser.load_occupancy_data1,
               'occupancy2' : data_parser.load_occupancy_data2,
               'occupancy3' : data_parser.load_occupancy_data3,
               'spambase'   : data_parser.load_spambase
    }
    header, data, target = options[args.dataset]()
    
    if len(sys.argv)==2:
        kMNS()
        SPC()
        AMC()     
    else:
        options = {'kmeans'         : kMNS,
                   'spectral'       : SPC,
                   'agglomerative'  : AMC
        }
        options[args.method]()
    
def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    
    parser.add_argument('dataset', type=str, 
        help='which dataset to be loaded, options: for datatest.txt type \'occupancy1\', for datatest2.txt type \'occupancy2\', for datatraining.txt type \'occupancy3\', or type \'spambase\'')
    parser.add_argument('--method', type=str, 
        help='which method to be executed, options: \'kmeans\', \'spectral\', \'agglomerative\'', default='')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))