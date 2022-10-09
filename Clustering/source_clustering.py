
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn_extra.cluster import KMedoids

class CSM:
      'This class provide methods useful to derive interesting information from a PE matrix'
      def __init__(self,csm_matrix,seed=0):
            '''
            Inputs : 
            @csm_matrix : The matrix M s.t. M[i,j] is the probability of error obtained training on i and evaluating on j.

            Attributes derived:
            @regret_matrix : Derived from csm_matrix. It's the matrix M s.t. M[i,j]=Regret_{i,j} (defined in the paper)
            
            '''
            self.csm_matrix=csm_matrix
            self.N=len(csm_matrix)
            self.regret_matrix=csm_matrix-np.tile(np.diag(csm_matrix).reshape(-1,1),(self.N)).T
            self.seed=seed

      def regret_clustering(self,nb_iter=50,current_labels=None):
            '''
            Inputs:
            @nb_iter : maximum of iterations 
            @current_labels : initialization of the clustering provided by the user

            Aim:
            This method builds a clustering of the sources involved in the csm matrix.
            The core idea is to foster the creation of clusters with relevant representatives.
            A relevant representative being a source thanks to which we can generalize as much as possible on the other sources in its cluster training from it.

            The pseudo-code is inspired from the K-means algorithm and it is available in the folder "Pseudo_codes".
                    
            '''

            if current_labels is None:
                '''
                Intuitive initialization : 
                For each source, assign the label corresponding to the other source in the grid minimizing
                the regret with it.
                '''
  
                current_labels=(self.regret_matrix+np.diag(243*[50])).argmin(axis=0)

            for iter in range(nb_iter):

                current_representatives=[]
                
                for k in np.unique(current_labels):
                    order=np.where(current_labels==k)[0]
                    cluster=self.regret_matrix[order,:][:,order]
                    virtual_repr=cluster.max(axis=1).argmin()
                    current_representatives.append(order[virtual_repr])

                current_representatives=np.array(current_representatives)
                current_labels=(self.regret_matrix[current_representatives]).argmin(axis=0)

            #at the end of the for loop we derive for the last time the best representatives in each cluster built so far.

            current_representatives=[]
            for k in np.unique(current_labels):
                order=np.where(current_labels==k)[0]
                cluster=self.regret_matrix[order,:][:,order]
                virtual_repr=cluster.max(axis=1).argmin()
                current_representatives.append(order[virtual_repr])

            current_representatives=np.array(current_representatives)

            return current_representatives,current_labels

      def k_medioid_clustering(self,K,seed=None):
         '''
         Inputs:
         @K : number of clusters wanted by the user
         @seed : The seed for the initialization of labels assignment
        
         Aim:
         This method builds a clustering of the sources involved in the csm matrix.
         The core idea is to represent each source with the concatenation of the column and the line representing it in the 
         PE matrix and then, use these features for a k-medioid algorithm.        
         '''
         features=np.concatenate([self.csm_matrix,self.csm_matrix.T],axis=1)
         if not(seed is None):
            kmedoids = KMedoids(n_clusters=K, init='k-medoids++',random_state=seed,metric='l2').fit(features)
         else:
            kmedoids = KMedoids(n_clusters=K, init='k-medoids++',metric='l2').fit(features)

         return kmedoids.medoid_indices_,kmedoids.labels_

      def greedy_covering(self,epsilon=10):
          '''
          This is the clustering algorithm used for our experiments in the paper.

          Input:
          @epsilon : maximum regret accepted between a representative and the members of its cluster.

          Output :
          @greedy_covering : The covering obtained with the details in the shape of a dictionary where the keys are the clusters
          representatives and the values the sources they are covering.
          @representatives : The list of the representatives (linked to a number)
          @labels : The labels of each source enabling to assign to each source a cluster
        
          Aim:
          This method builds a clustering of the sources involved in the csm matrix.
          The core idea is to see the clustering problem as a set-covering problem and deduce a solution of this problem
          using a greedy algorithm. More details about this idea are given in the article.

          The pseudo-code is available in the article and in the folder "Pseudo_codes".

          '''  

          P={}
          greedy_covering={}
          not_already_included_in_the_greedy_covering=0
          size=[]

          for i in range(len(self.regret_matrix)):
            P[i]=np.where(self.regret_matrix[i]<epsilon)[0]
            P_size=len(P[i])
            size.append(P_size)
            not_already_included_in_the_greedy_covering+=P_size
       
          while not_already_included_in_the_greedy_covering>0:
              not_already_included_in_the_greedy_covering=0
              points_covered_by_unit_cost={} 

              for i in range(len(self.regret_matrix)):

                if len(P[i]):
                    points_covered_by_unit_cost[i]=len(P[i]) #/constant_cost

                else:
                    points_covered_by_unit_cost[i]=-100

              # we look for the representative covering the maximum number of sources (regret radius < epsilon) 
              k=np.argmax(list(points_covered_by_unit_cost.values()))
              greedy_covering[k]=np.array(list(set(P[k]).union({k}))) 
              #Note : Above, the representative is explicitly included in its own cluster to prevent that the algorithm
              #returns a covering where a representative is covered by an other representative.

              for i in range(len(self.regret_matrix)):
                  #all the sources already covered are deleted from the initial covering
                  P[i]=np.array(list(set(P[i])-set(greedy_covering[k]))) 
                  not_already_included_in_the_greedy_covering+=len(P[i])

          #Safe check : What is the maximum value of the maximum regrets between each representative and its members ? 
          #It should be lower than epsilon
          max_regrets=[]
          for cover in greedy_covering.keys():
              max_regrets.append(self.regret_matrix[cover,greedy_covering[cover]].max())

          print('Max max regrets : ',np.max(max_regrets))

          #The greedy algorithm used here has a theoretical guarantee giving us an idea about how far we are from an optimal covering
          print('Minimum number of sources for the optimal covering :', np.ceil(len(greedy_covering.keys())/harmonic_sum(max(size))))

          labels_assignement=np.zeros(len(self.regret_matrix))
          
          for j in greedy_covering.keys():
              for value in greedy_covering[j]:
                    labels_assignement[value]=j

    
          return greedy_covering,np.sort(np.unique(labels_assignement)),np.array(labels_assignement)
          
                
      def plot_matrix(self,order=None,matrix_type='regret',title='csm_matrix'):
          '''
          This method enables to save an heatmap representing the PE or Regret matrix reordered according to an order
          proposed by the user. By default the order is the identity.

          Inputs : 
          @order : An ordering of the sources in self.csm_matrix 
          @matrix_type : The kind of matrix you want to plot (by default it's the regret matrix)
          @title : The title of the plot you are going to create
          
          '''

          order = np.arange(0,len(self.csm_matrix)) if order is None else order
          reordered_matrix=self.regret_matrix[order,:][:,order] if (matrix_type.lower()=='regret') else self.csm_matrix[order,:][:,order]

          num_ticks = len(reordered_matrix)

          # the index of the position of yticks
          yticks = np.linspace(0, num_ticks - 1, num_ticks,dtype=int)
          # the content of labels of these yticks
          yticklabels = [order[idx] for idx in yticks]

          plt.figure(figsize=(num_ticks,num_ticks))
          sns.heatmap(reordered_matrix, annot=True,cmap="flare",vmin=reordered_matrix.min(),vmax=reordered_matrix.max(),
          yticklabels=yticklabels,xticklabels=yticklabels)
          plt.savefig(f'{title}.pdf')
          plt.close()
        
        
        
