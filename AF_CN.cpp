#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <iomanip>

using namespace std;

double S_function(double x, double y){
    return (M_PI * M_PI) / 2.0 * cos(M_PI / 2.0 * x) * cos(M_PI / 2.0 * y);
}

double u_exact(double x, double y, double t) {
    return (1 - exp(- M_PI * M_PI / 2.0 * t)) * cos(M_PI * x / 2.0) * cos(M_PI * y / 2.0);
}

void ThomasSolve(const std::vector<double>& a,
                 const std::vector<double>& b,
                 const std::vector<double>& c,
                 const std::vector<double>& d,
                 std::vector<double>& x)
{
    int n = (int)b.size();
    if (n == 0) return;

    std::vector<double> c_star(n-1);
    std::vector<double> d_star(n);

    c_star[0] = c[0] / b[0];
    d_star[0] = d[0] / b[0];

    for (int i = 1; i < n-1; ++i) {
        double denom = b[i] - a[i-1] * c_star[i-1];
        c_star[i] = c[i] / denom;
        d_star[i] = (d[i] - a[i-1] * d_star[i-1]) / denom;
    }

    if (n > 1) {
        double denom_last = b[n-1] - a[n-2] * c_star[n-2];
        d_star[n-1] = (d[n-1] - a[n-2] * d_star[n-2]) / denom_last;
    }

    x[n-1] = d_star[n-1];
    for (int i = n-2; i >= 0; --i) {
        x[i] = d_star[i] - c_star[i] * x[i+1];
    }
}

int main(){
    double ax = -1.0, bx = 1.0;
    double ay = -1.0, by = 1.0;
    vector<double> h_values;
    vector<double> dt_values;
    vector<double> errs;
    vector<double> errs_dt;
    vector<double> cvgs_dt;
    vector<double> cvgs;
    for (int trial = 0; trial < 4; ++trial){
        double h = 1.0 / pow(2.0, (trial+7));
        h_values.push_back(h);
        if (trial == 0){
            for (int k = 0; k < 4; ++k){
                double dt = 1.0 / (100.0 * pow(2.0, k));
                dt_values.push_back(dt);
                double T = 0.1;
                int Nt = int(T / dt);

                int Nx = int((bx - ax) / h) + 1;
                int Ny = int((by - ay) / h) + 1;

                int n_inner_x = Nx - 2;
                int n_inner_y = Ny - 2;

                cout << "Nx=" << Nx << " Ny=" << Ny << " Nt=" << Nt << " h=" << h << " dt=" << dt << endl;

                vector<double> u(Nx * Ny, 0.0);
                vector<double> u_star(Nx * Ny, 0.0);
                vector<double> u_new(Nx * Ny, 0.0);
                vector<vector<double>> u_total;
                u_total.push_back(u);

                double rx = dt / (2.0 * h * h);
                double ry = rx;

                vector<double> a_x(n_inner_x-1, -rx);
                vector<double> b_x(n_inner_x, 1.0 + 2.0 * rx);
                vector<double> c_x(n_inner_x-1, -rx);

                vector<double> a_y(n_inner_y-1, -ry);
                vector<double> b_y(n_inner_y, 1.0 + 2.0 * ry);
                vector<double> c_y(n_inner_y-1, -ry);

                vector<double> rhs_row(n_inner_x), sol_row(n_inner_x);
                vector<double> rhs_col(n_inner_y), sol_col(n_inner_y);

                for (int step = 0; step < Nt; ++step) {
                    for (int j = 1; j <= Ny-2; ++j) {
                        for (int i = 1; i <= Nx-2; ++i) {
                            double uij = u[i + j * Nx];
                            double u_ip1j = u[(i+1) + j * Nx];
                            double u_im1j = u[(i-1) + j * Nx];

                            double u_ijp1 = u[i + (j+1) * Nx];
                            double u_ijm1 = u[i + (j-1) * Nx];

                            double rhs_val = uij + (dt * 0.5) * (u_ijp1 - 2.0 * uij + u_ijm1) / (h*h);

                            double xi = ax + i * h;
                            double yj = ay + j * h;
                            rhs_val += 0.5 * dt * S_function(xi, yj);

                            rhs_row[i-1] = rhs_val;
                        }

                        if (n_inner_x > 0) {

                            ThomasSolve(a_x, b_x, c_x, rhs_row, sol_row);

                            for (int i = 1; i <= Nx-2; ++i) {
                                u_star[i + j * Nx] = sol_row[i-1];
                            }
                        }
                        u_star[0 + j * Nx] = 0.0;
                        u_star[(Nx-1) + j * Nx] = 0.0;
                    }
                    for (int i = 0; i < Nx; ++i) {
                        u_star[i + 0 * Nx] = 0.0;
                        u_star[i + (Ny-1) * Nx] = 0.0;
                    }

                    for (int i = 1; i <= Nx-2; ++i) {
                        for (int j = 1; j <= Ny-2; ++j) {
                            double utij = u_star[i + j * Nx];
                            double ut_ip1j = u_star[(i+1) + j * Nx];
                            double ut_im1j = u_star[(i-1) + j * Nx];

                            double rhs_val = utij + (dt * 0.5) * (ut_ip1j - 2.0 * utij + ut_im1j) / (h*h);

                            double xi = ax + i * h;
                            double yj = ay + j * h;
                            rhs_val += 0.5 * dt * S_function(xi, yj);

                            rhs_col[j-1] = rhs_val;
                        }

                        if (n_inner_y > 0) {
                            ThomasSolve(a_y, b_y, c_y, rhs_col, sol_col);

                            for (int j = 1; j <= Ny-2; ++j) {
                                u_new[i + j * Nx] = sol_col[j-1];
                            }
                        }
                        u_new[i + 0 * Nx] = 0.0;
                        u_new[i + (Ny-1) * Nx] = 0.0;
                    }

                    for (int j = 0; j < Ny; ++j) {
                        u_new[0 + j * Nx] = 0.0;
                        u_new[(Nx-1) + j * Nx] = 0.0;
                    }
                    u.swap(u_new);
                    u_total.push_back(u);
                }

                for (int step = 0; step < Nt; ++step) {
                    double t = step * dt;
                    double L2_sum = 0.0;
                    double L_max = 0.0;
                    for (int j = 0; j <= Ny-1; ++j) {
                        for (int i = 0; i <= Nx-1; ++i) {
                            double x = ax + i * h;
                            double y = ay + j * h;
                            double t = step * dt;
                            double diffr = abs(u_total[step][i + j*Nx] -  u_exact(x, y, t));
                            L2_sum += diffr * diffr;
                            L_max = max(L_max, diffr);
                        }
                    }
                    double L2 = h*sqrt(L2_sum);

                    if (step == Nt-1){
                        errs_dt.push_back(L2);
                        cout << "L2_error: " << L2 << endl;
                        cout << "L_infinite: " << L_max << endl;
                    }
                }

                if (k > 0) {
                    double cvg = log(errs_dt[k-1]/errs_dt[k]) / log(dt_values[k-1]/dt_values[k]);
                    cvgs_dt.push_back(cvg);
                    cout << "CVG: " << cvg << endl;
                }
            }
        }

        double dt = 1.0 / 100.0;
        double T = 0.1;
        int Nt = int(T / dt);

        int Nx = int((bx - ax) / h) + 1;
        int Ny = int((by - ay) / h) + 1;

        int n_inner_x = Nx - 2;
        int n_inner_y = Ny - 2;

        cout << "Nx=" << Nx << " Ny=" << Ny << " Nt=" << Nt << " h=" << h << " dt=" << dt << endl;

        vector<double> u(Nx * Ny, 0.0);
        vector<double> u_star(Nx * Ny, 0.0);
        vector<double> u_new(Nx * Ny, 0.0);
        vector<vector<double>> u_total;
        u_total.push_back(u);

        double rx = dt / (2.0 * h * h);
        double ry = rx;

        vector<double> a_x(n_inner_x-1, -rx);
        vector<double> b_x(n_inner_x, 1.0 + 2.0 * rx);
        vector<double> c_x(n_inner_x-1, -rx);

        vector<double> a_y(n_inner_y-1, -ry);
        vector<double> b_y(n_inner_y, 1.0 + 2.0 * ry);
        vector<double> c_y(n_inner_y-1, -ry);

        vector<double> rhs_row(n_inner_x), sol_row(n_inner_x);
        vector<double> rhs_col(n_inner_y), sol_col(n_inner_y);

        for (int step = 0; step < Nt; ++step) {
            for (int j = 1; j <= Ny-2; ++j) {
                for (int i = 1; i <= Nx-2; ++i) {
                    double uij = u[i + j * Nx];
                    double u_ip1j = u[(i+1) + j * Nx];
                    double u_im1j = u[(i-1) + j * Nx];

                    double u_ijp1 = u[i + (j+1) * Nx];
                    double u_ijm1 = u[i + (j-1) * Nx];

                    double rhs_val = uij + (dt * 0.5) * (u_ijp1 - 2.0 * uij + u_ijm1) / (h*h);

                    double xi = ax + i * h;
                    double yj = ay + j * h;
                    rhs_val += 0.5 * dt * S_function(xi, yj);

                    rhs_row[i-1] = rhs_val;
                }

                if (n_inner_x > 0) {

                    ThomasSolve(a_x, b_x, c_x, rhs_row, sol_row);

                    for (int i = 1; i <= Nx-2; ++i) {
                        u_star[i + j * Nx] = sol_row[i-1];
                    }
                }
                u_star[0 + j * Nx] = 0.0;
                u_star[(Nx-1) + j * Nx] = 0.0;
            }
            for (int i = 0; i < Nx; ++i) {
                u_star[i + 0 * Nx] = 0.0;
                u_star[i + (Ny-1) * Nx] = 0.0;
            }

            for (int i = 1; i <= Nx-2; ++i) {
                for (int j = 1; j <= Ny-2; ++j) {
                    double utij = u_star[i + j * Nx];
                    double ut_ip1j = u_star[(i+1) + j * Nx];
                    double ut_im1j = u_star[(i-1) + j * Nx];

                    double rhs_val = utij + (dt * 0.5) * (ut_ip1j - 2.0 * utij + ut_im1j) / (h*h);

                    double xi = ax + i * h;
                    double yj = ay + j * h;
                    rhs_val += 0.5 * dt * S_function(xi, yj);

                    rhs_col[j-1] = rhs_val;
                }

                if (n_inner_y > 0) {
                    ThomasSolve(a_y, b_y, c_y, rhs_col, sol_col);

                    for (int j = 1; j <= Ny-2; ++j) {
                        u_new[i + j * Nx] = sol_col[j-1];
                    }
                }
                u_new[i + 0 * Nx] = 0.0;
                u_new[i + (Ny-1) * Nx] = 0.0;
            }

            for (int j = 0; j < Ny; ++j) {
                u_new[0 + j * Nx] = 0.0;
                u_new[(Nx-1) + j * Nx] = 0.0;
            }
            u.swap(u_new);
            u_total.push_back(u);
        }

        ofstream fout_exact("exact_u.txt");
        ofstream fout_num("numerical_u.txt");
        ofstream fout_err("error_u.txt");

        if (trial == 0) {
            double L2_sum = 0.0;
            double L_max = 0.0;
            for (int j = 0; j <= Ny-1; ++j) {
                for (int i = 0; i <= Nx-1; ++i) {
                    double x = ax + i * h;
                    double y = ay + j * h;
                    double t = (Nt-1)*dt;
                    double diffr = abs(u_total[Nt-1][i + j*Nx] -  u_exact(x, y, t));
                    L2_sum += diffr * diffr;
                    L_max = max(L_max, diffr);
                    fout_exact << x << " " << y << " " << u_exact(x, y, t) << "\n";
                    fout_num   << x << " " << y << " " << u_total[Nt-1][i + j*Nx] << "\n";
                    fout_err   << x << " " << y << " " << diffr << "\n";
                }
            }
        }

        for (int step = 0; step < Nt; ++step) {
            double t = step * dt;
            double L2_sum = 0.0;
            double L_max = 0.0;
            for (int j = 0; j <= Ny-1; ++j) {
                for (int i = 0; i <= Nx-1; ++i) {
                    double x = ax + i * h;
                    double y = ay + j * h;
                    double diffr = abs(u_total[step][i + j*Nx] -  u_exact(x, y, t));
                    L2_sum += diffr * diffr;
                    L_max = max(L_max, diffr);  
                }
            }
            double L2 = h*sqrt(L2_sum);

            if (step == Nt-1){
                errs.push_back(L2);
                cout << "L2_error: " << L2 << endl;
                cout << "L_infinite: " << L_max << endl;
            }
        }

        if (trial > 0) {
            double cvg = log(errs[trial-1]/errs[trial]) / log(h_values[trial-1]/h_values[trial]);
            cvgs.push_back(cvg);
            cout << "CVG: " << cvg << endl;
        }
    }
    return 0;
}
