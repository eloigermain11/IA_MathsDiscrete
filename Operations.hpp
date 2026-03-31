#ifndef OPERATIONS_HPP
#define OPERATIONS_HPP

#include "Tensor.hpp"
#include <memory>

class Operations {
public:
    static std::shared_ptr<Tensor> add(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    static std::shared_ptr<Tensor> matmul(std::shared_ptr<Tensor> a, std::shared_ptr<Tensor> b);
    static std::shared_ptr<Tensor> relu(std::shared_ptr<Tensor> a);
    // ... ajoute les autres ici (sigmoid, softmax, etc.)
};

#endif