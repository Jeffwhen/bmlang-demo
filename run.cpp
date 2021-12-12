#include <bmlib_runtime.h>
#include <bmruntime_interface.h>
#include <cstdlib>
#include <cmath>
#include <string>
#include <vector>
#include <iostream>

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
    if (argc != 2)
    {
        std::cerr << argv[0] << " <.bmodel>" << std::endl;
        return 1;
    }

    const char *model_path = argv[1];
    const int q_size = 1;
    const int dims = 512;
    const int default_db_size = 1024;
    const int top_num = 10;
	const int seed = 42;
    srand(seed);

    std::vector<int8_t> q_data(q_size * dims);
    std::vector<int8_t> db_data(dims * default_db_size);
    for (int i = 0; i < default_db_size; ++i)
        rand_feature(db_data.data() + i, dims, default_db_size);

    // select q_data
    const int sel = 42;
    for (int i = 0; i < dims; ++i)
        q_data[i] = db_data[sel + i * default_db_size];

#define call(fn, ...) \
    do { \
        auto ret = fn(__VA_ARGS__); \
        if (ret != BM_SUCCESS) \
        { \
            std::cerr << #fn << " failed " << ret << std::endl; \
            return -1; \
        } \
    } while (false);

    bm_handle_t handle;
    const int devid = 0;
    call(bm_dev_request, &handle, devid);
    bm_device_mem_t q, db, products, index;
    // Alloc input device memories and copy data
    call(bm_malloc_device_byte, handle, &q, q_data.size());
    call(bm_memcpy_s2d, handle, q, q_data.data());
    call(bm_malloc_device_byte, handle, &db, db_data.size());
    call(bm_memcpy_s2d, handle, db, db_data.data());
    // Output memories
    call(bm_malloc_device_byte, handle, &products, q_size * top_num * sizeof(float));
    call(bm_malloc_device_byte, handle, &index, q_size * top_num * sizeof(int32_t));

    void *bmrt = bmrt_create(handle);
    if (!bmrt_load_bmodel(bmrt, model_path))
    {
        std::cerr << "failed to load \""
                  << model_path << "\"" << std::endl;
        return 1;
    }

    // Prepare shapes
    bm_shape_t q_shape, db_shape, product_shape, index_shape;
    q_shape.num_dims = 2;
    q_shape.dims[0] = q_size;
    q_shape.dims[1] = dims;
    db_shape.num_dims = 2;
    db_shape.dims[0] = dims;
    db_shape.dims[1] = default_db_size;
    product_shape.num_dims = 2;
    product_shape.dims[0] = q_size;
    product_shape.dims[1] = top_num;
    index_shape.num_dims = 2;
    index_shape.dims[0] = q_size;
    index_shape.dims[1] = top_num;

    // Prepare tensors
    bm_tensor_t q_tensor, db_tensor, product_tensor, index_tensor;
    db_tensor.dtype = BM_INT8;
    db_tensor.shape = db_shape;
    db_tensor.device_mem = db;
    db_tensor.st_mode = BM_STORE_1N;

    q_tensor.dtype = BM_INT8;
    q_tensor.shape = q_shape;
    q_tensor.device_mem = q;
    q_tensor.st_mode = BM_STORE_1N;

    product_tensor.dtype = BM_FLOAT32;
    product_tensor.shape = product_shape;
    product_tensor.device_mem = products;
    product_tensor.st_mode = BM_STORE_1N;

    index_tensor.dtype = BM_INT32;
    index_tensor.shape = index_shape;
    index_tensor.device_mem = index;
    index_tensor.st_mode = BM_STORE_1N;

    // Inference
    const char **network_names;
    bmrt_get_network_names(bmrt, &network_names);
    const char *net_name = network_names[0];
    std::cout << "=================="
              << net_name
              << "=================="
              << std::endl;
    const int input_num = 2;
    const int output_num = 2;
    bm_tensor_t input_tensors[] = {q_tensor, db_tensor};
    bm_tensor_t output_tensors[] = {product_tensor, index_tensor};
    const bool user_mem = true;
    const bool user_st_mode = false;
    if (!bmrt_launch_tensor_ex(
        bmrt, net_name, input_tensors, input_num,
        output_tensors, output_num, user_mem, user_st_mode))
    {
        std::cerr << "failed to launch network" << std::endl;
        return -1;
    }
    call(bm_thread_sync, handle);

    std::vector<float> product_data(q_size * top_num);
    std::vector<int32_t> index_data(q_size * top_num);
    call(bm_memcpy_d2s, handle, product_data.data(), products);
    call(bm_memcpy_d2s, handle, index_data.data(), index);
    for (int i = 0; i < top_num; ++i)
    {
        std::cout << index_data[i] << " "
                  << product_data[i] / (127 * 127)
                  << std::endl;
    }

    bmrt_destroy(bmrt);
    bm_free_device(handle, q);
    bm_free_device(handle, db);
    bm_free_device(handle, products);
    bm_free_device(handle, index);
    bm_dev_free(handle);

    return 0;
}
