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
