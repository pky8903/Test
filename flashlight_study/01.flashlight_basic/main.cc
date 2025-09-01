#include <flashlight/fl/flashlight.h>
#include <iostream>

int main() {
    auto a = fl::Variable(
        fl::rand({4, 4}, fl::dtype::f32), true 
        );
    std::cout << "A " << a.tensor() << std::endl;
    
    auto b = fl::Variable(
        fl::rand({4, 4}, fl::dtype::f32), true
        );
    std::cout << "B " << b.tensor() << std::endl;

    auto ab = a * b;
    std::cout << "AB " << ab.tensor() << std::endl;

    ab.backward();
    std::cout << "AB backward " << ab.grad().tensor() << std::endl;

    std::cout << "A backward " << a.grad().tensor() << std::endl;

    std::cout << "B backward " << b.grad().tensor() << std::endl;

    return 0;
}

