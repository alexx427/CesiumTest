#include <fstream>
#include <vector>
#include <string>

#include "thrust/device_vector.h"

using namespace std;

template<int N = 0>
__global__ void downsample2(int * __restrict sbuf, int * __restrict dbuf, int sz)
{
    extern __shared__ int buf[];
    auto *bufb = (uint8_t*)buf;

    int wl = blockDim.x * 2;

    int ib2 = blockDim.x * blockIdx.x * 2;
    int il = threadIdx.x;

    int jl2wl = threadIdx.y * wl * 2;
    int j2sz = (blockDim.y*blockIdx.y + threadIdx.y) * 2 * sz;

    // Preload source rectangle using coalesced memory access
    buf[jl2wl + il] = sbuf[j2sz + ib2 + il];
    buf[jl2wl + blockDim.x + il] = sbuf[j2sz + ib2 + blockDim.x + il];
    buf[jl2wl + wl + il] = sbuf[j2sz + sz + ib2 + il];
    buf[jl2wl + wl + blockDim.x + il] = sbuf[j2sz + sz + ib2 + blockDim.x + il];
    __syncthreads();

    auto offl = jl2wl * 4;
    int res = 0;
    // Calculate averages of byte color components and merge them into int32
    for(int k = 0; k < 4; ++k)
    {
        // For output image sizes 2x2 and 1x1, last thread in line process N components only
        if constexpr(N != 0)
        {
            if(il == blockDim.x-1 && k >= N)
                break;
        }
        auto f = il*4+k;
        auto off = offl + f/3*6 + f%3;
        res |= ((int(bufb[off]) + int(bufb[off+3]) + int(bufb[off+wl*4]) + int(bufb[off+wl*4+3])) / 4) << (k*8);
    }

    auto *dst = &dbuf[(j2sz/2 + ib2)/2 + il];
    if constexpr(N == 2)
    {
        // for output size 2x2 first thread in line writes 4 bytes and second writes 2 bytes
        if(il == 1)
            *((int16_t*)dst) = int16_t(res);
        else
            *dst = res;
    }
    else
        *dst = res;
}

void saveImage(const thrust::device_vector<int32_t> &deviceBuf, int mipmapNum, int tsize)
{
    auto tsize4 = max(1,tsize/4);
    vector<int32_t> buf(tsize4);
    thrust::copy(deviceBuf.begin(), deviceBuf.begin() + tsize4, buf.begin());

    ofstream f("data/vancouver_"s + to_string(mipmapNum) + ".data", ios::binary);
    f.write((char*)buf.data(), tsize);
}

int main()
{
    constexpr int inputImageSize = 8192;
    ifstream f("data/vancouver.data", ios::binary);
    if(!f.is_open())
    {
        cerr << "Cannot find input file!\n";
        return -1;
    }

    vector<int32_t> buf(inputImageSize * inputImageSize * 3 / 4);
    f.read((char*)buf.data(), buf.size()*4);

    thrust::device_vector<int32_t> srcBuf(buf.begin(), buf.end());
    thrust::device_vector<int32_t> dstBuf(srcBuf.size()/4);

    unsigned tsize = 0, mipmapNum = 0;
    for(unsigned i = inputImageSize/2; i > 0; i /= 2)
    {
        auto blocky = min(32u,i);
        dim3 threads {blocky >= 4 ? blocky/4*3 : blocky, blocky, 1};
        dim3 blocks {i/blocky, i/blocky, 1};

        if(i > 2)
            downsample2<<<blocks, threads, threads.x * threads.y * 4 * sizeof(int32_t)>>>(srcBuf.data().get(), dstBuf.data().get(), i*3/2);
        else if(i > 1)
            downsample2<2><<<blocks, threads, threads.x * threads.y * 4 * sizeof(int32_t)>>>(srcBuf.data().get(), dstBuf.data().get(), i*3/2);
        else
            downsample2<3><<<blocks, threads, threads.x * threads.y * 4 * sizeof(int32_t)>>>(srcBuf.data().get(), dstBuf.data().get(), i*3/2);

        // Overlap on-device new mipmap calculation with previous mipmap saving
        if(tsize != 0)
            saveImage(srcBuf, mipmapNum, tsize);

        cudaDeviceSynchronize();

        srcBuf.swap(dstBuf);
        tsize = i*i*3;
        ++mipmapNum;
    }
    saveImage(srcBuf, mipmapNum, tsize);

    return 0;
}
