#include <flashlight/fl/flashlight.h>
#include <flashlight/fl/tensor/Index.h>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace fl;

Variable cropCenterRegion(const Variable& var, const int min_width, const int min_height)
{
    if (static_cast<int>(var.dim(0)) == min_width 
        and static_cast<int>(var.dim(1)) == min_height) 
    {
        return var.asContiguous();     
    }
    auto seq_0 = range(var.dim(0)/2 - min_width/2, var.dim(0)/2 + min_width/2);
    auto seq_1 = range(var.dim(1)/2 - min_height/2, var.dim(1)/2 + min_height/2);
    return var({seq_0, seq_1});
}

std::vector<Variable> fitToMinSize(const std::vector<Variable>& vars) {
    if (vars.empty()) {
      return {};
    }
    
    int min_width = vars[0].dim(0);
    int min_height = vars[0].dim(1);
    
    for (const auto& var : vars) {
      min_width = std::min(min_width, static_cast<int>(var.dim(0)));
      min_height = std::min(min_height, static_cast<int>(var.dim(1)));
    }
    
    std::vector<Variable> result;
    result.reserve(vars.size());
    
    std::transform(vars.begin(), vars.end(), std::back_inserter(result), [min_width, min_height](const Variable& var) {
            return cropCenterRegion(var, min_width, min_height);
        }
    ); 
    
    return result;
}

void test(int size) {
    auto a = Variable(
        rand({size, size, 1, 1}, dtype::f32), true 
        );
    
    auto b = Variable(
        rand({size + 4, size + 4, 1, 1}, dtype::f32), true
        );

    auto c = Variable(
        rand({size + 2, size + 2, 1, 1}, dtype::f32), true
        );

    auto cropped = fitToMinSize({a, b, c}); 

    auto gamma = Variable(
        rand({size, size, 1, 1}, dtype::f32), false
        );

    auto scaled = std::vector<Variable>(cropped.size());

    std::transform(cropped.begin(), cropped.end(), std::begin(scaled), [&gamma](const Variable& var) {
            return var * gamma;
        }
    );

    auto reduced = scaled[0] + scaled[1] + scaled[2];

    auto scalar = fl::sum(reduced, {0, 1, 2, 3}, true);
    scalar.backward();
}

int main() {
    for (auto i = 0; i < 1; ++i) {
        test(2048);
    }
    return 0;
}

