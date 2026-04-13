#ifndef TENSOR_HPP
#define TENSOR_HPP

#include <vector>
#include <memory>
#include <functional>
#include <string>

class Tensor {
public:
    // Valeurs stockees
    std::vector<float> donnees; // valeurs actuelle numeriques (weights ou mots etc)
    std::vector<float> gradient; // valeurs de l'erreur accumulée
    std::vector<int> dimensions; // shape du tensor

    // Gradient
    std::vector<std::shared_ptr<Tensor>> parents_precedents; // pointeur vers les parents
    std::function<void()> fonction_backpropagation; // lambda / backpropagation, c genre le chemin du retour

    // methodes
    Tensor(std::vector<int> dimensions_souhaitees); // initialisateur
    void initialiser_gradient_a_zero(); // remet gradients a 0
    void lancer_backpropagation(); // calcule les erreurs en remontant le graphe
};

#endif



