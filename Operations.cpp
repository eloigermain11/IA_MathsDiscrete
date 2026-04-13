// Auteur   : Éloi Germain
// Date     : 31/03/2026
// Sujet    : Operations sur vecteurs et tenseurs

#include "Operations.hpp"
#include <cmath>
#include <stdexcept>

// les fonctions calculent la valeur, l'erreur et le gradient


// Addition
std::shared_ptr<Tensor> Operations::addition(std::shared_ptr<Tensor> tenseur_a, std::shared_ptr<Tensor> tenseur_b) {
    
    // Vérifier la shape des matrices pour s'Assurer que l'addition est possible
    if (tenseur_a->dimensions != tenseur_b->dimensions) {
        throw std::runtime_error("Erreur : Les dimensions des tenseurs doivent être identiques pour l'addition.");
    }

    // Création du tenseur output, qui a la meme taille que ses parents
    auto resultat = std::make_shared<Tensor>(tenseur_a->dimensions);

    // Forward Pass (somme de tout les données case par case)
    for (size_t i = 0; i < tenseur_a->donnees.size(); ++i) {
        resultat->donnees[i] = tenseur_a->donnees[i] + tenseur_b->donnees[i];
    }

    // Graphe de calcul
    // Garde en memoire les parents pour pouvoir remonter plus tard
    resultat->parents_precedents = {tenseur_a, tenseur_b};

    // Le Backward Pass (gradient) fonction lambda en memoire (calcule l'erreur)
    resultat->fonction_backpropagation = [tenseur_a, tenseur_b, resultat]() {
    
        for (size_t index = 0; index < resultat->gradient.size(); ++index) {
            
            // Reupere l'erreur accumulee dans le resultat
            float erreur_entrante = resultat->gradient[index];

            // Transmet cette erreur au premier parent (tenseur_a)
            // Derivee est 1 faque lerreur reste identique
            tenseur_a->gradient[index] += erreur_entrante * 1.0f;

            // Transmet la même erreur au deuxième parent (tenseur_b)
            tenseur_b->gradient[index] += erreur_entrante * 1.0f;
        }
    };


    return resultat;
}

// Multiplication matricielle (MatMul)
std::shared_ptr<Tensor> Operations::multiplication_matricielle(std::shared_ptr<Tensor> tenseur_a, std::shared_ptr<Tensor> tenseur_b) {
    
    // Verifier dimensions
    int nombre_lignes_a = tenseur_a->dimensions[0];
    int nombre_commun = tenseur_a->dimensions[1];
    int nombre_colonnes_b = tenseur_b->dimensions[1];

    if (tenseur_a->dimensions[1] != tenseur_b->dimensions[0]) {
        throw std::runtime_error("Dimensions incompatibles pour la multiplication matricielle.");
    }

    // Résultat (M x N)
    auto resultat = std::make_shared<Tensor>(std::vector<int>{nombre_lignes_a, nombre_colonnes_b});
    // Forward Pass (Triple boucle for), produit scalaire
    for (int ligne = 0; ligne < nombre_lignes_a; ++ligne) {
        for (int colonne = 0; colonne < nombre_colonnes_b; ++colonne) {
            float somme_accumulee = 0.0f;
            for (int k = 0; k < nombre_commun; ++k) {
                // On multiplie la ligne de A par la colonne de B
                somme_accumulee += tenseur_a->donnees[ligne * nombre_commun + k] * tenseur_b->donnees[k * nombre_colonnes_b + colonne];
            }
            resultat->donnees[ligne * nombre_colonnes_b + colonne] = somme_accumulee;
        }
    }
    

    // Backpropagation pour MatMul
    resultat->parents_precedents = {tenseur_a, tenseur_b};
    
    resultat->fonction_backpropagation = [tenseur_a, tenseur_b, resultat, nombre_lignes_a, nombre_commun, nombre_colonnes_b]() {
    
    // Calcul du gradient pour le tenseur_a
    // Formule : Gradient_A = Gradient_Resultat * Transposée(Tenseur_B)
    // Dimensions : (M x N) * (N x K) = (M x K)
    for (int ligne = 0; ligne < nombre_lignes_a; ++ligne) {
        for (int k = 0; k < nombre_commun; ++k) {
            float accumulation_erreur = 0.0f;
            for (int colonne = 0; colonne < nombre_colonnes_b; ++colonne) {
                // On utilise la valeur de B pour pondérer l'erreur qui revient
                float valeur_tenseur_b = tenseur_b->donnees[k * nombre_colonnes_b + colonne];
                float erreur_provenant_du_resultat = resultat->gradient[ligne * nombre_colonnes_b + colonne];
                
                accumulation_erreur += erreur_provenant_du_resultat * valeur_tenseur_b;
            }
            // On ajoute l'erreur accumulée au gradient existant de A
            tenseur_a->gradient[ligne * nombre_commun + k] += accumulation_erreur;
        }
    }

    // Calcul du gradient pour le tenseur_b
    // Formule : Gradient_B = Transposée(Tenseur_A) * Gradient_Resultat
    // Dimensions : (K x M) * (M x N) = (K x N)
    for (int k = 0; k < nombre_commun; ++k) {
        for (int colonne = 0; colonne < nombre_colonnes_b; ++colonne) {
            float accumulation_erreur = 0.0f;
            for (int ligne = 0; ligne < nombre_lignes_a; ++ligne) {
                // On utilise la valeur de A pour pondérer l'erreur qui revient
                float valeur_tenseur_a = tenseur_a->donnees[ligne * nombre_commun + k];
                float erreur_provenant_du_resultat = resultat->gradient[ligne * nombre_colonnes_b + colonne];
                
                accumulation_erreur += valeur_tenseur_a * erreur_provenant_du_resultat;
            }
            // On ajoute l'erreur accumulée au gradient existant de B
            tenseur_b->gradient[k * nombre_colonnes_b + colonne] += accumulation_erreur;
        }
    }
};

    return resultat;
}


// Fonction ReLU
std::shared_ptr<Tensor> Operations::relu(std::shared_ptr<Tensor> tenseur_entree) {
    auto resultat = std::make_shared<Tensor>(tenseur_entree->dimensions);

    // Forward Pass
    for (size_t i = 0; i < tenseur_entree->donnees.size(); ++i) {
        // Si positif on la garde sinon on met 0
        resultat->donnees[i] = (tenseur_entree->donnees[i] > 0.0f) ? tenseur_entree->donnees[i] : 0.0f;
    }

    resultat->parents_precedents = {tenseur_entree};

    resultat->fonction_backpropagation = [tenseur_entree, resultat]() {
        for (size_t i = 0; i < resultat->gradient.size(); ++i) {
            // Si la donnée originale était positive, l'erreur passe. 
            // Sinon, l'erreur est bloquée (multipliée par 0).
            float pente = (tenseur_entree->donnees[i] > 0.0f) ? 1.0f : 0.0f;
            tenseur_entree->gradient[i] += resultat->gradient[i] * pente;
        }
    };

    return resultat;
}






