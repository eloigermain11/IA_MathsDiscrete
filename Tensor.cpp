#include "Tensor.hpp"
#include <algorithm>
#include <numeric>

// constructeur de tensor
Tensor::Tensor(std::vector<int> dimensions_souhaitees) {
    dimensions = dimensions_souhaitees;
    
    // calcul de la taille quon a besoin en memoire (ex tensor 3x4 = 12 cases en memoire)
    size_t nombre_elements_total = 1;
    for (int dimension_individuelle : dimensions) {
        nombre_elements_total *= dimension_individuelle;
    }

    // allocation de la memoire, pour les donnees et le gradient, et mets a 0
    donnees.assign(nombre_elements_total, 0.0f);
    gradient.assign(nombre_elements_total, 0.0f);
}

// nettoyer les erreurs precedantes
void Tensor::initialiser_gradient_a_zero() {
    std::fill(gradient.begin(), gradient.end(), 0.0f);
}



// Partie mathématique: parcours de graphe dirigé
// Chaque tensor est un sommet, chaque parents_precedents est un edge
// ca donne un graphe acylique dirigé (DAG)

// backpropagation
void Tensor::lancer_backpropagation() {
    
    bool gradient_est_vide = true;

    // initialiser lerreur de depart (si fonction commence sur la derniere valeur, gradient = 1)
    for (float valeur_gradient : gradient) {
        if (valeur_gradient != 0.0f) {
            gradient_est_vide = false;
            break;
        }
    }
    if (gradient_est_vide) {
        std::fill(gradient.begin(), gradient.end(), 1.0f);
    }

    // calcul local: distribue l'erreur vers les parents
    if (fonction_backpropagation) {
        fonction_backpropagation();
    }

    // recursion (graphe) demande a chaque parent de faire la meme chose, remonte tout le graphe
    for (auto& parent_precedent : parents_precedents) {
        parent_precedent->lancer_backpropagation();
    }
}