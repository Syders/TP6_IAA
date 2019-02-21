import numpy as np
import time

from bayes import GaussianBayes
from utils import load_dataset, plot_scatter_hist

#main
#Ouverture du fichier
nomFichier="data2.csv"
data_info, data_labels = load_dataset("./data/"+nomFichier)

#Separation des datas selon un pourcentage
num_data=data_info.shape[0]
num_feature=data_info.shape[1]

data_pourc=0.80

num_learn=int(num_data*data_pourc)
num_test=num_data-num_learn

train_data=np.ndarray(shape(num_learn,num_feature),dtype=float)
train_labels=np.ndarray(shape(num_learn),dtype=int)

test_data=np.ndarray(shape(num_test,num_feature),dtype=float)
test_label=np.ndarray(shape(num_test),dtype=int)

n_classe=len(np.unique(data_labels))
for i in range(n_classes):
    data_classe = [data_info[j] for j in range(n_data) if (i==data_labels[j])]
    num_data_classe=data_classe.shape[0]
    num_learn_classe=int(num_data_classe*data_pourc)
    train_data.append(data_classe[:num_learn_classe])
    train_labels.append(data_classe[:num_learn_classe])

#print("num_data:" + str(num_data)+ "\nnum_learn:" + str(num_learn))
train_data=data_info[:num_learn]
train_labels=data_labels[:num_learn]
#print("train_data.shape="+str(train_data.shape[0]))
test_data=data_info[num_learn:]
test_label=data_labels[num_learn:]
#print("test_data.shape="+str(test_data.shape[0]))

#Traiter les données

#Bayes
# affichage
plot_scatter_hist(train_data, train_labels)
label=np.zeros(len(np.unique(data_labels)))
nb_data=train_data.shape[0]
for i in range(nb_data):
    prior[train_labels[i]]+=1/nb_data

print("Prior:=" ,prior, "\n")
        
print("Gaussian:\n")
# Instanciation de la classe GaussianB
g = GaussianBayes(priors=prior,optim=True)

# Apprentissage
start=time.time()
g.fit(train_data, train_labels)
end=time.time()
print("Execution time of Gaussian Learn:" + str(end-start))

#Score 
scoreGaussian=g.score(test_data,test_label)
print("Score gaussian=" + str(scoreGaussian))



