#include <iostream>
#include <iomanip>
#include "Tensor.hpp"
#include "Operations.hpp"

// juste pour afficher un tenseur
void afficher_tenseur(const std::string& nom, std::shared_ptr<Tensor> tenseur, bool afficher_gradient = false) {
    std::cout << nom << ": " << std::endl;
    int colonnes = tenseur->dimensions[1];
    
    for (size_t i = 0; i < tenseur->donnees.size(); ++i) {
        // choix entre afficher la valeur ou le gradient
        float valeur = afficher_gradient ? tenseur->gradient[i] : tenseur->donnees[i];
        std::cout << std::fixed << std::setprecision(2) << valeur << "  ";
        
        // retour à la ligne pour rendre ca propre
        if ((i + 1) % colonnes == 0) std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    // creer les donnees d'entree
    // matrice de weights A et de donnee B (2x2 pour l'instant)
    auto tenseur_a = std::make_shared<Tensor>(std::vector<int>{2, 2});
    auto tenseur_b = std::make_shared<Tensor>(std::vector<int>{2, 2});
    auto tenseur_biais = std::make_shared<Tensor>(std::vector<int>{2, 2});

    // valeurs test
    tenseur_a->donnees = {1.0f, 2.0f, 3.0f, 4.0f};
    tenseur_b->donnees = {0.5f, -1.0f, 1.0f, 0.0f};
    tenseur_biais->donnees = {1.0f, 1.0f, 1.0f, 1.0f};

    std::cout << "FORWARD PASS**********************************" << std::endl;

    // Multiplication Matricielle (A * B)
    // Résultat attendu [[2.5, -1.0], [5.5, -3.0]]
    auto resultat_multiplication = Operations::multiplication_matricielle(tenseur_a, tenseur_b);

    // Addition du Biais ( (A*B) + Biais )
    // Résultat attendu [[3.5, 0.0], [6.5, -2.0]]
    auto resultat_addition = Operations::addition(resultat_multiplication, tenseur_biais);

    // ReLU (coupe tout ce qui est négatif)
    // Résultat attendu [[3.5, 0.0], [6.5, 0.0]]
    auto resultat_final = Operations::relu(resultat_addition);

    // Affiche pour vérifier
    afficher_tenseur("Entree A", tenseur_a);
    afficher_tenseur("Entree B", tenseur_b);
    afficher_tenseur("Resultat Final (apres ReLU)", resultat_final);

    std::cout << "BACKWARD PASS *********************************" << std::endl;

    // lance backpropagation, part du dernier tenseur et on remonte le graphe
    resultat_final->lancer_backpropagation();

    // verifier les gradients
    // Le gradient de A doit nous dire comment modifier A pour changer le résultat final.
    afficher_tenseur("Gradient de A", tenseur_a, true);
    afficher_tenseur("Gradient de B", tenseur_b, true);
    afficher_tenseur("Gradient du Biais", tenseur_biais, true);

    return 0;
}