#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>
#include <cusparse.h>

using namespace std;

__device__ double S_function_dev(double x, double y) {
    const double PI = 3.14159265358979323846;
    return (PI * PI) / 2.0 * cos(PI * 0.5 * x) * cos(PI * 0.5 * y);
}

__device__ double u_exact_dev(double x, double y, double t) {
    const double PI = 3.14159265358979323846;
    return (1.0 - exp(- (PI * PI / 2.0) * t))
         * cos(PI * x * 0.5) * cos(PI * y * 0.5);
}

__global__ void d_rhs_val_x(double dt, double h, int Nx, int Ny,
                            const double* __restrict__ d_u,
                            double* __restrict__ d_rhs)
{
    int j_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (j_idx < 1 || j_idx > Ny - 2) return;

    double yj = -1.0 + j_idx * h;
    int Nx_inner = Nx - 2;
    int j_inner  = j_idx - 1;

    for (int i = 1; i < Nx - 1; ++i) {
        int idx = i + Nx * j_idx;
        double xi = -1.0 + h * i;

        double rhs;
        if (i == 1) {
            rhs = d_u[idx]
                + 0.5 * dt * (-2.0 * d_u[idx] + d_u[idx + 1]) / (h * h);
        } else if (i == Nx - 2) {
            rhs = d_u[idx]
                + 0.5 * dt * (d_u[idx - 1] - 2.0 * d_u[idx]) / (h * h);
        } else {
            rhs = d_u[idx]
                + 0.5 * dt * (d_u[idx - 1] - 2.0 * d_u[idx] + d_u[idx + 1]) / (h * h);
        }
        rhs += 0.5 * dt * S_function_dev(xi, yj);

        int i_inner   = i - 1;
        int inner_idx = j_inner * Nx_inner + i_inner;
        d_rhs[inner_idx] = rhs;
    }
}

__global__ void d_rhs_val_y(double dt, double h, int Nx, int Ny,
                            const double* __restrict__ d_u,
                            double* __restrict__ d_rhs_y)
{
    int i_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (i_idx < 1 || i_idx > Nx - 2) return;

    double xi = -1.0 + i_idx * h;
    int Nx_inner = Nx - 2;
    int Ny_inner = Ny - 2;
    int i_inner  = i_idx - 1;

    for (int j = 1; j < Ny - 1; ++j) {
        int idx = i_idx * Ny + j;
        double yj = -1.0 + h * j;

        double rhs;
        if (j == 1) {
            rhs = d_u[idx]
                + 0.5 * dt * (-2.0 * d_u[idx] + d_u[idx + 1]) / (h * h);
        } else if (j == Ny - 2) {
            rhs = d_u[idx]
                + 0.5 * dt * (d_u[idx - 1] - 2.0 * d_u[idx]) / (h * h);
        } else {
            rhs = d_u[idx]
                + 0.5 * dt * (d_u[idx - 1] - 2.0 * d_u[idx] + d_u[idx + 1]) / (h * h);
        }
        rhs += 0.5 * dt * S_function_dev(xi, yj);

        int j_inner   = j - 1;
        int inner_idx = i_inner * Ny_inner + j_inner;
        d_rhs_y[inner_idx] = rhs;
    }
}

__global__ void update_full_u(int Nx, int Ny,
                              double* __restrict__ d_u,
                              const double* __restrict__ d_rhs_inner)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int Nx_inner = Nx - 2;
    int Ny_inner = Ny - 2;
    if (idx >= Nx_inner * Ny_inner) return;

    int j_inner = idx / Nx_inner;
    int i_inner = idx % Nx_inner;

    int i = i_inner + 1;
    int j = j_inner + 1;
    int full_idx = j * Nx + i;

    d_u[full_idx] = d_rhs_inner[idx];
}

__global__ void transpose_inner_ytox(double* d_rhs_y, double* d_u, int Nx, int Ny){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N_inner = (Nx-2) * (Ny-2);
    if (idx >= N_inner) return;

    int j_inner = idx / (Ny-2);
    int i_inner = idx % (Ny-2);

    int i = i_inner + 1;
    int j = j_inner + 1;
    int full_idx = j * Nx + i;
    d_u[full_idx] = d_rhs_y[idx];
}

__global__ void transpose_inner_xtoy(double* d_rhs, double* d_u_star, int Nx, int Ny){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N_inner = (Nx-2) * (Ny-2);
    if (idx >= N_inner) return;

    int j_inner = idx / (Nx-2);
    int i_inner = idx % (Nx-2);

    int i = i_inner + 1;
    int j = j_inner + 1;
    int full_idx = i * Ny + j;
    d_u_star[full_idx] = d_rhs[idx];
}

__global__ void compute_error_kernel(double ax, double ay,
                                     double h, double t,
                                     int Nx, int Ny,
                                     const double* __restrict__ d_u,
                                     double* __restrict__ d_err2)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int Ntot = Nx * Ny;
    if (idx >= Ntot) return;

    int i = idx % Nx;
    int j = idx / Nx;

    double x = ax + i * h;
    double y = ay + j * h;

    double u_ex = u_exact_dev(x, y, t);
    double diff = d_u[idx] - u_ex;
    d_err2[idx] = diff * diff;
}

int main() {
    double ax = -1.0, bx = 1.0;
    double ay = -1.0, by = 1.0;
    std::vector<double> h_values;
    std::vector<double> errs;

    for (int trial = 0; trial < 2; ++trial) {

        double h  = 1.0 / pow(2.0, trial + 9);
        h_values.push_back(h);
        double dt = 1.0 / 100.0;
        double T  = 4.0;
        int Nt = int(T / dt) + 1;

        int Nx = int((bx - ax) / h) + 1;
        int Ny = int((by - ay) / h) + 1;

        int n_inner_x = Nx - 2;
        int n_inner_y = Ny - 2;

        cout << "Nx=" << Nx << " Ny=" << Ny << " Nt=" << Nt
             << " h=" << h << " dt=" << dt << endl;

        vector<double> u_host(Nx * Ny, 0.0);

        double rx = dt / (2.0 * h * h);
        double ry = rx;

        vector<double> a_x(n_inner_x, -rx);
        vector<double> b_x(n_inner_x, 1.0 + 2.0 * rx);
        vector<double> c_x(n_inner_x, -rx);
        a_x[0] = 0.0;
        c_x[n_inner_x - 1] = 0.0;

        vector<double> a_y(n_inner_y, -ry);
        vector<double> b_y(n_inner_y, 1.0 + 2.0 * ry);
        vector<double> c_y(n_inner_y, -ry);
        a_y[0] = 0.0;
        c_y[n_inner_y - 1] = 0.0;

        double *d_u, *d_u_star, *d_u_new;
        double *d_rhs, *d_rhs_y;
        double *d_a_x, *d_b_x, *d_c_x;
        double *d_a_y, *d_b_y, *d_c_y;
        double *d_err2;

        size_t sz = Nx * Ny * sizeof(double);

        cudaMalloc(&d_u,      sz);
        cudaMalloc(&d_u_star, sz);
        cudaMalloc(&d_u_new,  sz);
        cudaMalloc(&d_rhs,   (size_t)(Nx - 2) * (Ny - 2) * sizeof(double));
        cudaMalloc(&d_rhs_y, (size_t)(Ny - 2) * (Nx - 2) * sizeof(double));
        cudaMalloc(&d_err2,  sz);

        cudaMalloc(&d_a_x, n_inner_x * sizeof(double));
        cudaMalloc(&d_b_x, n_inner_x * sizeof(double));
        cudaMalloc(&d_c_x, n_inner_x * sizeof(double));

        cudaMalloc(&d_a_y, n_inner_y * sizeof(double));
        cudaMalloc(&d_b_y, n_inner_y * sizeof(double));
        cudaMalloc(&d_c_y, n_inner_y * sizeof(double));

        
        cudaEvent_t ev_htod_start, ev_htod_stop, ev_dtoh_start, ev_dtoh_stop;
        cudaEventCreate(&ev_htod_start);
        cudaEventCreate(&ev_htod_stop);
        cudaEventCreate(&ev_dtoh_start);
        cudaEventCreate(&ev_dtoh_stop);

        cudaEventRecord(ev_htod_start);
        cudaMemcpy(d_u, u_host.data(), sz, cudaMemcpyHostToDevice);
        cudaMemcpy(d_u_star, u_host.data(), sz, cudaMemcpyHostToDevice);

        cudaMemcpy(d_a_x, a_x.data(), n_inner_x * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_x, b_x.data(), n_inner_x * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c_x, c_x.data(), n_inner_x * sizeof(double), cudaMemcpyHostToDevice);

        cudaMemcpy(d_a_y, a_y.data(), n_inner_y * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b_y, b_y.data(), n_inner_y * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c_y, c_y.data(), n_inner_y * sizeof(double), cudaMemcpyHostToDevice);
        cudaEventRecord(ev_htod_stop);
        cudaEventSynchronize(ev_htod_stop);

        float htod_ms = 0.0f;
        cudaEventElapsedTime(&htod_ms, ev_htod_start, ev_htod_stop);
        double htod_sec = htod_ms * 1e-3;

        cudaDeviceSynchronize();

        cusparseHandle_t handle;
        cusparseCreate(&handle);

        int threads_per_block = 512;
        int num_block_x    = (Nx + threads_per_block - 1) / threads_per_block;
        int num_block_y    = (Ny + threads_per_block - 1) / threads_per_block;
        int num_block_full = (Nx * Ny + threads_per_block - 1) / threads_per_block;

        int system_size_x = Nx - 2;
        int batchcount_y  = Ny - 2;
        int system_size_y = Ny - 2;
        int batchcount_x  = Nx - 2;


        size_t bufferSize_x = 0;
        cusparseDgtsv2_bufferSizeExt(
            handle, system_size_x, batchcount_y, d_a_x, d_b_x, d_c_x,
            d_rhs, system_size_x,
            &bufferSize_x
        );
        void* buffer_x;
        cudaMalloc(&buffer_x, bufferSize_x);

        size_t bufferSize_y = 0;
        cusparseDgtsv2_bufferSizeExt(
            handle, system_size_y, batchcount_x, d_a_y, d_b_y, d_c_y,
            d_rhs_y, system_size_y,
            &bufferSize_y
        );
        void* buffer_y;
        cudaMalloc(&buffer_y, bufferSize_y);

        int device;
        cudaGetDevice(&device);
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);

        int numBlocksPerSM = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &numBlocksPerSM,
            d_rhs_val_x,
            threads_per_block,
            0
        );
        float occupancy =
            (numBlocksPerSM * threads_per_block) /
            (float)prop.maxThreadsPerMultiProcessor;

        cudaEvent_t ev_start, ev_stop;
        cudaEventCreate(&ev_start);
        cudaEventCreate(&ev_stop);

        double t_rhs_x_sec   = 0.0;
        double t_solve_x_sec = 0.0;
        double t_rhs_y_sec   = 0.0;
        double t_solve_y_sec = 0.0;
        double t_update_sec  = 0.0;

        auto start_total = chrono::high_resolution_clock::now();

        for (int step = 0; step < Nt - 1; ++step) {

            cudaEventRecord(ev_start);
            d_rhs_val_x<<<num_block_y, threads_per_block>>>(
                dt, h, Nx, Ny, d_u, d_rhs
            );
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            {
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_start, ev_stop);
                t_rhs_x_sec += ms * 1e-3;
            }

            cudaEventRecord(ev_start);
            cusparseDgtsv2(
                handle,
                system_size_x, batchcount_y,
                d_a_x, d_b_x, d_c_x,
                d_rhs,
                system_size_x,
                buffer_x
            );
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            {
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_start, ev_stop);
                t_solve_x_sec += ms * 1e-3;
            }

            cudaEventRecord(ev_start);
            transpose_inner_xtoy<<<num_block_full, threads_per_block>>>(d_rhs, d_u_star, Nx, Ny);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            {
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_start, ev_stop);
                t_update_sec += ms * 1e-3;
            }

            cudaEventRecord(ev_start);
            d_rhs_val_y<<<num_block_x, threads_per_block>>>(
                dt, h, Nx, Ny, d_u_star, d_rhs_y
            );
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            {
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_start, ev_stop);
                t_rhs_y_sec += ms * 1e-3;
            }

            cudaEventRecord(ev_start);
            cusparseDgtsv2(
                handle,
                system_size_y, batchcount_x,
                d_a_y, d_b_y, d_c_y,
                d_rhs_y,
                system_size_y,
                buffer_y
            );
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            {
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_start, ev_stop);
                t_solve_y_sec += ms * 1e-3;
            }

            cudaEventRecord(ev_start);
            transpose_inner_ytox<<<num_block_full, threads_per_block>>>(d_rhs_y, d_u, Nx, Ny);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            {
                float ms = 0.0f;
                cudaEventElapsedTime(&ms, ev_start, ev_stop);
                t_update_sec += ms * 1e-3;
            }
        }

        auto end_total = chrono::high_resolution_clock::now();
        double wall_seconds = chrono::duration<double>(end_total - start_total).count();

        double t_final = (Nt - 1) * dt;
        int num_block_err = num_block_full;

        compute_error_kernel<<<num_block_err, threads_per_block>>>(
            ax, ay, h, t_final, Nx, Ny, d_u, d_err2
        );
        cudaDeviceSynchronize();

        vector<double> err2_host(Nx * Ny);

        cudaEventRecord(ev_dtoh_start);
        cudaMemcpy(err2_host.data(), d_err2, sz, cudaMemcpyDeviceToHost);
        cudaEventRecord(ev_dtoh_stop);
        cudaEventSynchronize(ev_dtoh_stop);
        float dtoh_ms = 0.0f;
        cudaEventElapsedTime(&dtoh_ms, ev_dtoh_start, ev_dtoh_stop);
        double dtoh_sec = dtoh_ms * 1e-3;

        double L2_sum = 0.0;
        double L_max  = 0.0;
        for (int j = 0; j < Ny; ++j) {
            for (int i = 0; i < Nx; ++i) {
                int idx = i + j * Nx;
                double diff_sq = err2_host[idx];
                L2_sum += diff_sq;
                double diff = std::sqrt(diff_sq);
                if (diff > L_max) L_max = diff;
            }
        }
        double L2 = h * std::sqrt(L2_sum);
        errs.push_back(L2);

        double Nx_inner = Nx - 2;
        double Ny_inner = Ny - 2;
        double bytes_rhs = 4.0 * Nx_inner * Ny_inner * sizeof(double); // per sweep
        double steps = double(Nt - 1);

        double rhs_x_time_per_step = t_rhs_x_sec / steps;
        double rhs_y_time_per_step = t_rhs_y_sec / steps;

        double rhs_x_bw = bytes_rhs / rhs_x_time_per_step / 1e9; // GB/s
        double rhs_y_bw = bytes_rhs / rhs_y_time_per_step / 1e9; // GB/s

        cout << "Host->Device transfer (s) = " << htod_sec << endl;
        cout << "Device->Host transfer (s) = " << dtoh_sec << endl;
        cout << "Elapsed wall time (s)     = " << wall_seconds << endl;
        cout << "RHS x total time (s)      = " << t_rhs_x_sec   << endl;
        cout << "Solve x total time (s)    = " << t_solve_x_sec << endl;
        cout << "RHS y total time (s)      = " << t_rhs_y_sec   << endl;
        cout << "Solve y total time (s)    = " << t_solve_y_sec << endl;
        cout << "Update total time (s)     = " << t_update_sec  << endl;
        cout << "Approx. BW (RHS x)        = " << rhs_x_bw      << " GB/s" << endl;
        cout << "Approx. BW (RHS y)        = " << rhs_y_bw      << " GB/s" << endl;
        cout << "Theoretical occupancy     = " << occupancy * 100.0f << " %" << endl;
        cout << "Final L2 error            = " << L2   << endl;
        cout << "Final L_inf error         = " << L_max << endl;

        if (trial > 0) {
            double cvr = log(errs[trial] / errs[trial - 1]) / log(h_values[trial] / h_values[trial-1]);
            cout << "Convergenc Rate = " << cvr << endl;
        }

        cudaEventDestroy(ev_start);
        cudaEventDestroy(ev_stop);
        cudaEventDestroy(ev_htod_start);
        cudaEventDestroy(ev_htod_stop);
        cudaEventDestroy(ev_dtoh_start);
        cudaEventDestroy(ev_dtoh_stop);

        cudaFree(buffer_x);
        cudaFree(buffer_y);

        cudaFree(d_u);
        cudaFree(d_u_star);
        cudaFree(d_u_new);
        cudaFree(d_rhs);
        cudaFree(d_rhs_y);
        cudaFree(d_err2);
        cudaFree(d_a_x);
        cudaFree(d_b_x);
        cudaFree(d_c_x);
        cudaFree(d_a_y);
        cudaFree(d_b_y);
        cudaFree(d_c_y);
        cusparseDestroy(handle);
    }

    return 0;
}
