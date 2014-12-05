#ifndef TVAL3_floatS_H_
#define TVAL3_floatS_H_

#include <stdexcept>
#include <string>

struct tval3_options {
	float mu; // maximum value for penalty parameter mu, see (2.12)
	          // mu is mainly decided by noise level. Set mu big when b is noise-free,
	          // set mu small when b is very noisy
	          // suggested values for mu: 2⁴..2¹³
	float mu0; // initial value for mu
	float beta; // maximum value for penalty parameter beta, see (2.12)
	float beta0; // initial value for beta
	float rate_cnt; // continuation parameter for both mu and beta
	float tol; // outer loop tolerance
	float tol_inn; // inner loop tolerance
	int maxcnt; // maximum number of outer iterations
	int maxit; // maximum number of iterations (total)
	float c; // corresponds to the parameter delta in (2.33) (Armijo condition)
	float gamma; // corresponds to rho "Algorithm 2" on page 29
	float gam; // corresponds to parameter eta in (2.34)
	           // Control the degree of nonmonotonicity. 0 corresponds to monotone line search.
	           // The best convergence is obtained by using values closer to 1 when the iterates
	           // are far from the optimum, and using values closer to 0 when near an optimum.
	float rate_gam; // shrinkage rate of gam
	bool isreal;
	bool nonneg;
	tval3_options() : mu(256.0f), mu0(256.0f), beta(32.0f), beta0(32.0f),
		rate_cnt(2.0f),
		tol(1e-6f), tol_inn(1e-3f), maxcnt(10), maxit(1025), c(1e-5f),
		gamma(0.6f),
		gam(0.9995f), rate_gam(0.9f), isreal(true), nonneg(true) {
	};
};

struct tval3_info {
	int total_iters;
	int outer_iters;
	float rel_chg;
	float rel_error;
	double secs;
};

class tval3_gpu_exception : public std::runtime_error {
	public:
		tval3_gpu_exception(const std::string& message) : std::runtime_error(message) 
	{ };
};

#endif /* TVAL3_floatS_H_ */
