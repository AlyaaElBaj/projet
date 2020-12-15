# Projet final Machine Learning
### Construction d'un modèle d'apprentissage profond qui prédit le volume du trafic.
Les données utilisées dans ce projet sont téléchargées de Kaggle "Radar Traffic Data. Traffic data collected from radar sensors deployed by the City of Austin".

## Présentation de l'ensemble des données : 
Dans le cadre d'un projet de modélisation prédictive, il est important de comprendre notre ensemble de données. J'aimerais commencer par expliquer ce qu'est le "volume de trafic", qui est la dernière colonne de notre ensemble de données et qui en serait le résultat. Le volume du trafic est défini comme le nombre de véhicules qui passent par un point d'une installation de transport pendant une période de temps donnée, qui est généralement d'une heure. 
L'ensemble de données comprend 4 603 861 lignes et 12 colonnes : **location_name**, **location_latitude**, **location_longitude**, **Year**, **Month**, **Day**, **Day of Week**, **Hour**, **Minute**, **Time Bin**, **Direction**, **Volume**

Ci-dessous un aperçu sur les données:
![data](https://user-images.githubusercontent.com/72867518/102202614-d5202600-3ec7-11eb-98df-f3208b417d31.JPG)
De façon globale nous pouvons regrouper les informations que nous présente la base de données comme suivant:
- ** La localisation ** : est donnée par les trois premières colonnes où on voit apparaître le nom de la localisation en plus de sa latitude et longitude. Ainsi que la colonne 'Direction' avec les possibilités suivantes: 'NB','SB','EB','WB', 'None'.
- ** La date ** : est donnée par 7 colones à savoir, 'Year','Month', 'Day', 'Day of Week', 'Hour', Minute', 'Time Bin'.
- ** Le traffic **: est donné par la dernière colonne volume et qui est notre variable qu'on cherche à prédire.

Ci-dessous quelques informations complémentaires sur notre Data Set:

![infodata](https://user-images.githubusercontent.com/72867518/102204160-c8043680-3ec9-11eb-818a-dd4e4897ebe9.JPG)

Pour résumer, notons que notre data set s'étale sur une période de 3 ans allant de 2017 à 2019, et qu'il comporte 32 couple (location, direction).

#### Problématique:
Notre but dans ce qui suit c'est de prédire le traffic c'est à dire le 'Volume' par heure pour chaque couple (location,direction) pour chaque jour sur une période de temps précise. Ainsi, j'ai commencé tout d'abords par transformer mes données en série temporelle pour chaque couple (location,direction). Dans ce qui suit, je présenterai plus en détails le preprocessing des données, puis la construction du modèle avec les choix des paramètres et finalement l'analyse des résultats obtenues.

## Data Preprocessing:
J'ai commencé d'abord par supprimer les colonnes dont je n'aurai pas besoin, à savoir:
- 'location_latitude' 'location_longitude' car j'utiliserai seuleument 'location_name' pour la prédiction.
- ' Time Bin' car c'est déjà donné par 'Hour' et 'Minute'.
- 'Day of Week' car je n'utiliserai pas cette information puisque la série temporelle que je contruit est avec une unité de temps de 1 heure par jour.
-  Je construierai une colonne pour la date sous la forme "%Y%m%d%H%M" par exemple "2017-07-26 14:00:00" puis je supprimerai les colonnes: 'Year','Month', 'Day', 'Day of Week', 'Hour', Minute'.

Il faut noter qu'il n'y a pas de Nan de notre data set, cependant, j'ai décidé de supprimer les lignes où la direction est donnée par None.

Après modifications, notre data set prêt pour le modèle est donné par:
![newdata](https://user-images.githubusercontent.com/72867518/102208655-f9800080-3ecf-11eb-83d7-8d1c2b85a280.JPG)

J'ai construis par la suite une fonction qui prend un couple (location,direction) et retourne sa série temporelle correspondante. Je me suis ensuite servis de cette fonction pour construire un dictionnaire qui prend comme clé le couple (location,direction) et lui attribue sa série temporelle.

Le but étant donc de préparer notre série temporelle pour le modèle et avoir suffisamment d'échantillons pour la phase du training.

Dans ce qui suit, j'ai utilisé un Convolutionnal Neural Networks qui est souvant utilisé dans le cas de prédiction pour les séries temporelles. Notons aussi que cette approche a été inspirée des explications de Monsieur Christophe Cerisara pour la construction d'un réseau de neuronnes convolutionnel pour la prédiction de ventes.

## Le model: Convolutionnal Neural Networks - CNN

J'expliquerai en détails dans cette partie la démarche utilisée, la structure ainsi que la mise en place des paramètres du modèle.

#### Sliding Window:
Comme mentionné précédemment, j'ai commencé par créer des échantillons avec leurs labels. La prédiction sera basé sur les 4 mois précédent (le input est donc de taille 24*30*4) pour prédire le traffic pour chaque jour de la semaine qui suit (Le output est donc de taille 24*7). Notre fenêtre glissante va donc avancer avec un pas de temps égal à la taille du output.

La fonction * ts_sequence_building * prend en argument une série temporelle de volume de trafic pour un couple (emplacement, direction) et applique une fenêtre de découpage (donc la taille d'avancement et le pas du temps) et renvoie une liste d'échantillons avec leurs labels. Ensuite, nous générons un train set et test set pour les inputs et outputs à partir des listes renvoyées par la fonction * ts_sequence_building *. Pour ce faire, j'ai utilisé la commande *train_test_split* .

#### Structure et mise en place du réseau de neuronnes:
J'ai utilisé un 1D Convolutionnal Neural Networks avec deux couches de convolution (avec kernel_size=3 pour les deux) et deux fully connected layers. Enfin, j'ai utilisé Adam optimizer avec un learning rate que je ferai varier pour voir s'il y a des améliorations et j'ai utilisé le MSE loss.

J'ai testé différentes valeurs pour le nombres de neuronnes (je détaillerai les résultats de chaque choix par la suite), et j'ai ajouté un dropout layer pour empêcher les problèmes du overfitting dans le modèle.

Je me suis servi du dictionnaire qui prend comme clé le couple (location,direction) et lui attribue sa série temporelle mentionné précedemment, afin d'entraîner et évaluer le model sur chaque couple avec les données correspondantes.

## Les résultats: analyse et comparaison
#### L'influence du nombre de neuronnes:
J'ai fixé tout d'abord dans ce qui suit le nombre d'epochs à 500, et le learning rate du Adam optimizer à 0.0005.

J'ai regardé pour les couples ('100 BLK S CONGRESS AVE (Congress Bridge)', 'NB') et ('100 BLK S CONGRESS AVE (Congress Bridge)', 'SB').
##### Premier cas avec maxpool2= 5 et out_features=80 pour fc1:
Dans ce cas, j'ai fixé le maxpool pour la deuxième couche convolutionnelle à 5, et le nombre de neuronnes pour le preumier fully connected layer à 80.

Pour le couple ('100 BLK S CONGRESS AVE (Congress Bridge)', 'NB'):

![neurone1](https://user-images.githubusercontent.com/72867518/102231868-bda86380-3eee-11eb-80f9-b802c41e2690.JPG)

![neuronnes1](https://user-images.githubusercontent.com/72867518/102225167-2db2eb80-3ee7-11eb-8ed9-ad2f53df7c41.png)

Pour le couple ('100 BLK S CONGRESS AVE (Congress Bridge)', 'SB'):

![neurone1bis](https://user-images.githubusercontent.com/72867518/102232414-62c33c00-3eef-11eb-95c0-87b83cd12e40.JPG)

![neuronnes1bis](https://user-images.githubusercontent.com/72867518/102232416-648cff80-3eef-11eb-9b0c-e1be385fdd70.png)


##### Deuxième cas avec maxpool2= 2 et out_features=50 pour fc1:
J'ai remarqué que le modèle prenait beaucoup de temps pour renvoyer les loss, j'ai donc décidé de diminuer le maxpool pour la deuxième couche convolutionnelle à 2, et le nombre de neuronnes pour le preumier fully connected layer à 50.

Pour le couple ('100 BLK S CONGRESS AVE (Congress Bridge)', 'NB'):

![neurone2](https://user-images.githubusercontent.com/72867518/102230163-d1eb6100-3eec-11eb-93d7-110b4a9808f9.JPG)

![neuronnes2](https://user-images.githubusercontent.com/72867518/102225878-ff81db80-3ee7-11eb-8257-ba6348c545cd.png)

Pour le couple ('100 BLK S CONGRESS AVE (Congress Bridge)', 'SB'):

![neurone2bis](https://user-images.githubusercontent.com/72867518/102230526-445c4100-3eed-11eb-92be-f315a66904b0.JPG)

![neuronnes2bis](https://user-images.githubusercontent.com/72867518/102230507-40302380-3eed-11eb-909f-5857e24764b6.png)


##### Troisième cas avec maxpool2= 8 et out_features=120 pour fc1:
J'ai décidé cette fois de rendre mon réseau de neuronnes un peu plus riche et compliqué, j'ai donc décidé d'augmenter le maxpool pour la deuxième couche convolutionnelle à 8, et le nombre de neuronnes pour le preumier fully connected layer à 130.

Pour le couple ('100 BLK S CONGRESS AVE (Congress Bridge)', 'NB'):

Les résultats obtenus sont les suivants:

![neurone3](https://user-images.githubusercontent.com/72867518/102232806-db29fd00-3eef-11eb-87c9-88b9d96da684.JPG)

![neuronnes3](https://user-images.githubusercontent.com/72867518/102232812-dcf3c080-3eef-11eb-8998-ba9f53fef455.png)

Pour le couple ('100 BLK S CONGRESS AVE (Congress Bridge)', 'SB'):

![neurone3bis](https://user-images.githubusercontent.com/72867518/102233504-a1a5c180-3ef0-11eb-9cce-277d3fa4af2f.JPG)

![neuronnes3bis](https://user-images.githubusercontent.com/72867518/102233513-a5394880-3ef0-11eb-99d6-af481ebe78b3.png)

##### Synthèse:


#### L'influence du nombre des epochs:

J'ai fixé dans ce cas le learning rate du Adam optimizer à 0.0005, le maxpool pour la deuxième couche convolutionnelle à 5, et le nombre de neuronnes pour le preumier fully connected layer à 80.

J'ai regardé pour le couple ('100 BLK S CONGRESS AVE (Congress Bridge)', 'NB') et ('100 BLK S CONGRESS AVE (Congress Bridge)', 'SB').

Regardons l'évolution des train loss et test loss entre 500 et 1500 epochs. 

Pour le couple ('100 BLK S CONGRESS AVE (Congress Bridge)', 'NB'):

![epochs1](https://user-images.githubusercontent.com/72867518/102238844-8342c480-3ef6-11eb-9869-92c9d02a352e.JPG)

![epochss1](https://user-images.githubusercontent.com/72867518/102238851-863db500-3ef6-11eb-84b9-5c623aa625d9.png)

Pour le couple ('100 BLK S CONGRESS AVE (Congress Bridge)', 'SB'):

![epochs2](https://user-images.githubusercontent.com/72867518/102239740-783c6400-3ef7-11eb-9d13-571b051a2f5d.JPG)

![epochss2](https://user-images.githubusercontent.com/72867518/102239756-7bcfeb00-3ef7-11eb-9e37-d2159acc781b.png)

##### Synthèse:


## Autres tentatives réalisées

Durant mon projet, j'ai essayé une autre approche avec une couche de convolution une couche d'activation non linéaire et un fully connected layer. La différence résidait aussi dans le splitting du train et test set. Quelques soucis techniques m'ont empêchés de me tarder sur ce modèle plutôt simple en comparaison avec le premier modèle et consacrer mes efforts à analyser plus en détails les résultats obtenus pour ce premier modèle.

## Suggéstions d'améliorations

Comme vous avez dû le remarquer, tout le travail de recherche d'un loss minimale était basé sur une recherche manuelle où je modifiais à la main les paramètres, répéter cette procédure jusqu'à trouver les bon paramètres permettant de miniminer le loss. Une méthode plus efficace serait d'avoir une partie du code qui fait le tunning des hyperparametres. A savoir par des méthodes d'optimisation.

## Les difficultés rencontrées et surmontées

La correction du modèle des réseaux de neuronnes convolutionnelles que nous avions vu avec Monsieur Christophe Cerisara pour les prédictions des ventes m'a beaucoup aidé pour assimiler différents concepts. Cependant, par manque de pratique et de manipulation, c'était un peu délicat de s'adapter avec l'environnement git. Après plusieurs tentatives, j'ai résussis à utiliser Git Desktop qui facilite la tâche. Il a aussi fallut s'adapter à la rédaction d'un rapport sur MarkDown, pour ce faire j'ai télécharger Barckets qui a une extention MarkDown et qui permet de visualiser facilement le texte. Je copiais ensuite ma rédation sur mon fichier sur Github.

## Les références

 [1] [Cours de Machine Learning de Monsieur Christophe Cerisara](https://members.loria.fr/CCerisara/#courses/machine_learning/) 
 
 [2] https://www.kaggle.com/hanjoonchoe/cnn-time-series-forecasting-with-pytorch
 
 [3] https://towardsdatascience.com/encoder-decoder-model-for-multistep-time-series-forecasting-using-pytorch-5d54c6af6e60


