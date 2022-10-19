import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
import yaml
from pathlib import Path

def harmonic_sum(n):
    i = 1
    s = 0.0
    for i in range(1, n+1):
        s = s + 1/i;
    return s;

def find_config(bayer_method,denoising,sharpen_micro,crop,ps_sharpening):  
    '''

    This function takes as input the config parameters of a pipeline and give back its number in our nomenclature.

    Inputs:
    @bayer_method : The Demosaicking method of the pipeline
    @denoising : The Denoising factor of the  pipeline
    @sharpen : The Micro-Sharpening factor of the  pipeline
    @crop : The Crop factor of the  pipeline
    @ps_sharpening : The Post-Resize-Sharpening factor of the pipeline

    Output:
    The number associated to the pipeline in our nomenclature

    '''
    denoising_to_match = denoising if denoising else "false"
    sharpen_micro_to_match = sharpen_micro if sharpen_micro else "false"
    crop_to_match = crop if crop else "false"
    ps_sharpening_to_match = ps_sharpening if ps_sharpening else "false"

    tuple_key = f"({bayer_method,denoising_to_match,sharpen_micro_to_match,crop_to_match,ps_sharpening_to_match})"
    config = yaml.safe_load(Path(f'config_to_code.yaml').read_text())    
    
    return config[tuple_key]

def build_dataset(labels):

    '''

    This function takes as input the labels returned by a certain clustering algorithm and gives back a dataset with the pipelines
    parameters as features and the corresponding labels as targets.

    Inputs:
    @labels : The labels returned by a certain clustering algorithm

    Output:
    @X : The features i.e. the pipeline parameters of each source
    @y : The labels associated to the features i.e. the input labels

    '''

    X=[]
    code_to_config = yaml.safe_load(Path(f'code_to_config.yaml').read_text())
    for i in range(243):
        X.append(code_to_config[i])
                            
    return np.array(X),labels

def parameter_importance(labels,title=None):
        '''
        This function plots the pipeline parameters importance in terms of mean decrease in impurity.
        The MDI measures the discriminative power of the parameters if we are using them to
        explain a source clustering obtained by a certain algorithm.

        Input : 
        @labels : The labels enabling to assign to each source, a cluster
        @title : Title of the plot saved
        
        '''
        index=['Demoisaicking','Denoising','SharpenMicro','DownSampling','PostResizeSharpening']
        index.reverse()
        
        X,y=build_dataset(labels)
        X_cat=np.zeros(shape=(len(labels),len(index)))
        le = preprocessing.LabelEncoder()

        for i in range(0,len(index)):
            X_cat[:,i]=le.fit_transform(X[:,i])
            
        clf = RandomForestClassifier(max_depth=10,criterion='entropy',n_estimators=100)
        clf.fit(X_cat,y)

        importances = clf.feature_importances_.tolist()
        importances.reverse()
        std = np.std([clf.feature_importances_ for tree in clf.estimators_], axis=0)
        
        fig, ax = plt.subplots(figsize=(6,3))
        ax.barh(index,importances,color='dodgerblue',alpha=0.8,edgecolor='black')
        ax.set_title("Parameters impact measured with mean decrease in impurity (MDI)",fontsize='10')
        ax.yaxis.set_label_position("right")
        ax.set_ylabel("MDI",rotation='270',labelpad = 50)
        fig.tight_layout()

        if not(title is None):
            plt.savefig(f'{title}.pdf', bbox_inches='tight')
      

universe = {
'Demosaicking': ['amaze','lmmse','fast'],
'Denoising': ['0','50','100'],
'SharpenMicro': ['0','50','100'],
'Downsampling' : ['0.15','0.25','1.0'],
'PostResizeSharpening': ['0','1.5','3']
}


def update_links_strength(links_strength,l):
    for i in range(1,len(l)):
        links_strength.setdefault(i,{})
        links_strength[i].setdefault(f'{l[i-1]}-{l[i]}',0)
        links_strength[i][f'{l[i-1]}-{l[i]}']+=1

def plot_lines_from_links(links_strength):
    
    for i in range(0,len(links_strength)):
        max_strength=max(links_strength[i+1].values())
        for link in links_strength[i+1]:
            current_strength=links_strength[i+1][link]
            color = 'mediumseagreen' if (current_strength==max_strength) else 'black'
            left=int(link[0])
            right=int(link[-1])
            plt.plot([6*(i+1),6*(i+2)],[left,right],c=color,lw=0.3*current_strength)
            
    
def cluster_to_graph(representative_number,labels,title=None):
    '''
    This function plot a network graph enabling to better understand the content of a cluster from a source clustering returned
    by a certain algorithm.

    Inputs:
    @representative_number: The number of the representative of the cluster we want to observe.
    @labels: The assignement of clusters for each source in the PE matrix
    @title: The title of the pdf file containing the plot generated by this function.
    
    '''
    X,y=build_dataset(labels)

    fig, ax = plt.subplots(figsize=(12,4))
    positions=[1,3,5]
    keys=list(universe.keys())
    links_strength={}

    for source in X[y==representative_number]:
        l=[]
        for key,param in zip(keys,source):
            l.append(positions[np.where(np.array(universe[key])==param)[0][0]])

        update_links_strength(links_strength,l)

    plot_lines_from_links(links_strength)

    representative=X[representative_number] 
    
    for i,key in zip(range(1,len(keys)+1),keys):
        values=universe[key]
        for k,value in enumerate(values):
            color ='tomato' if (value==representative[i-1]) else 'dodgerblue'
            text = value
  
            plt.annotate(
                text,
                xy=(6*i, positions[k]), xytext=(0,0),
                textcoords='offset points', ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.6', fc=color,alpha=1),
                arrowprops=dict(arrowstyle = '->', connectionstyle='arc3,rad=0'))

        plt.xlim(4,32)
        plt.ylim(0,6)

    ax.set_xticks(np.arange(6,(len(keys)+1)*6,6))
    ax.set_xticklabels(keys);
    ax.get_yaxis().set_visible(False)

    if not(title is None):
        plt.savefig(f'{title}.pdf', bbox_inches='tight')