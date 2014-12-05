#ifndef HANDLE_ERROR_H_
#define HANDLE_ERROR_H_

#include <stdexcept>
#include <string>
#include <sstream>

class cuda_exception : public std::runtime_error {
	public:
		cuda_exception(const std::string& message) : std::runtime_error(message) 
	{ };
};

static void handle_error(cudaError_t error, const char *file, int line ) {
	if (error != cudaSuccess) {
		std::stringstream ss;
		ss << file << ", line " << line << ": " << cudaGetErrorString(
		        error) << "\n";
		throw cuda_exception(ss.str());
	}
}

static void handle_error(cublasStatus_t error, const char *file, int line ) {
	if (error != CUBLAS_STATUS_SUCCESS) {
		std::stringstream ss;
		ss << file << ", line " << line << ": cublas error " << error <<
		        "\n";
		throw cuda_exception(ss.str());
	}
}

static void handle_error(cusparseStatus_t error, const char *file, int line ) {
	if (error != CUSPARSE_STATUS_SUCCESS) {
		std::stringstream ss;
		ss << file << ", line " << line << ": cusparse error " <<
		        error << "\n";
		throw cuda_exception(ss.str());
	}
}

#define HANDLE_ERROR(error) (handle_error(error, __FILE__, __LINE__ ))

#endif /* HANDLE_ERROR_H_ */
