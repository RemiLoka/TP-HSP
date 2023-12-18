# TP : Implémentation d'un CNN - LeNet-5 sur GPU

## Introduction

### Hardware utilisé

Pour ce TP, nous avons utilisé un GPU NVidia Quadro RTX4000. Utiliser un GPU NVidia est indispensable pour ce TP au vu du code en cuda.

### Avertissement

Sur ce GitHub, il n'y a pas les fichier compilé, uniquement ceux en .cu. Il est donc indispensable de les compiler.

Pour compiler:

` nvcc fichier.cu -o fichier_cuda `

Pour executer:

` ./fichier_cuda ` et avec timing ` time ./fichier_cuda `

## Partie 1 - Prise en main de Cuda : Multiplication de matrices

Le but de cette partie est de comparer les performances en exécutant sur le CPU et sur le GPU.
Pour se faire nous avons calculé sur CPU et GPU, l'addition et la multiplication de matrice.

### Pour l'exécution

Il est nécessaire pour l'exécution de préciser la taille de la matrice.
Par exemple ` ./fichier_cuda 1000 1000`

### Résultats

On a testé pour deux tailles de matrices différentes:

- Pour une matrice de taille 100x100:

  - Temps CPU addition: 61.0 µs
  - Temps GPU addition: 30.3 µs

  - Temps CPU mutilplication: 4.4 ms
  - Temps GPU multiplication: 33.7 µs

- Pour une matrice de taille 1000x1000:

  - Temps CPU addition: 5.3 ms
  - Temps GPU addition: 71.5 µs

  - Temps CPU mutilplication: 7.3 s
  - Temps GPU multiplication: 17.6 ms

### Comparaison des résultats

Lors des calculs sur des matrices de petites tailles, l'utilisation du GPU n'est pas indispensable. Mais des lors que la taille des matrices dépasse 1000x1000, il est indispensable d'utiliser le GPU, en effet, l'execution sur GPU est plus rapide d'un ordre 100.

## Partie 2 - Premières couches du réseau de neurone LeNet-5 : Convolution 2D et subsampling

Pour tester les Layers de cette partie, nous avons créer une fonction de test.

$$\[
\begin{bmatrix}
  a_{11} & a_{12} & a_{13} \\
  a_{21} & a_{22} & a_{23} \\
  a_{31} & a_{32} & a_{33} \\
\end{bmatrix}
\]$$

### Layer 1 - Génération des données de test

### Layer 2 - Layer 2 - Convolution 2D

### Layer 3 - Sous-échantillonnage

### Tests

### Fonctions d'activation

## Partie 3 - Un peu de Python
