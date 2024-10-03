#include <iostream>

#include "array.hpp"

using namespace nd;

int main()
{
    try
    {
        nd::array A({ 2, 2 });
        nd::array B({ 2, 2 });
        auto s = A.shape();

        float count = 1.0f;
        for (size_t c = 0; c < s[0]; ++c)
        {
            for (size_t r = 0; r < s[1]; ++r)
            {
                A({ c, r }) = count;
                B({ c, r }) = count + 4;
                count++;
            }
        }

        /*
        * 1 3   5 7
        * 2 4   6 8
        */

        auto C = A.concat(B);
        for (auto n : C)
        {
            std::cout << n << " ";
        }

        C = A.concat(B, 1);
        for (auto n : C)
        {
            std::cout << n << " ";
        }
    }
    catch (std::exception& e)
    {
        std::cerr << e.what() << std::endl;
    }
}