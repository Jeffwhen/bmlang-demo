#include <bmlang.h>

void rand_feature(int8_t *data, int dims, int stride = 1)
{
    std::vector<float> f(dims);
    float mod = 0;
    for (int i = 0; i < dims; i++)
    {
        float v = rand() * 1.f / RAND_MAX - 0.5;
        f[i] = v;
        mod += v * v;
    }
    mod = sqrt(mod);
    for (int i = 0; i < dims; i++)
        data[stride * i] = round((f[i] / mod) * 127.f);
}

int main(int argc, char *argv[])
{
    const int q_size = 1;
    const int dims = 512;
    const int default_db_size = 1024;

	const int seed = 42;
    srand(seed);
    bmlang::init(bmlang::BOTH, "BM1684");

    std::vector<int8_t> q_data(q_size * dims);
    std::vector<int8_t> db_data(dims * default_db_size);
    for (int i = 0; i < q_size; ++i)
        rand_feature(q_data.data(), dims);
    for (int i = 0; i < default_db_size; ++i)
        rand_feature(db_data.data() + i, dims, default_db_size);
    bmlang::Tensor q("q", bmlang::INT8, q_size, dims);
    bmlang::Tensor db("db", bmlang::INT8, dims, default_db_size);
    q.set_data(reinterpret_cast<const char *>(q_data.data()));
    db.set_data(reinterpret_cast<const char *>(db_data.data()));

    std::vector<bmlang::Tensor *> mul_input{&q, &db};
    bmlang::Tensor inner_products(bmlang::INT16);
    // do mat mul
    bmlang::matrix_mul(mul_input, inner_products);

    // topk only supports fp32
    // so cast it
    bmlang::Tensor fp_products(bmlang::FLOAT32);
    bmlang::cast(inner_products, fp_products);

    const int k = 10;
    const int axis = 1;
    bmlang::TopkParam param(k, axis);
    bmlang::Tensor sorted_products("sorted_products", bmlang::FLOAT32);
    bmlang::Tensor topk_index("topk_index", bmlang::INT32);
    // do topk
    bmlang::topk(fp_products, sorted_products, topk_index, param);

    std::vector<bmlang::Tensor *> input_tensors{&q, &db};
    std::vector<bmlang::Tensor *> ref_tensors{&sorted_products, &topk_index};
    const int opt_level = 2;
    const bool dynamic = false;
    const bool enable_profile = false;
    const char *name = "cosine";
    // generate model
    bmlang::compile_with_check(
        name, input_tensors, ref_tensors, opt_level, dynamic, enable_profile);

    bmlang::deinit();

    return 0;
}
