#pragma once
#include <stdlib.h>
#include <vector>

inline void rand_feature(int8_t *data, int dims, int stride = 1)
{
    std::vector<float> f(dims);
    float mod = 0;
    for (int i = 0; i < dims; i++)
    {
        float v = rand() * 1.f / RAND_MAX;
        f[i] = v;
        mod += v * v;
    }
    mod = sqrt(mod);
    for (int i = 0; i < dims; i++)
        data[i + stride] = round((f[i] / mod) * 127.f);
}
