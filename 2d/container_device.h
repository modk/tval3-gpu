#ifndef CONTAINER_DEVICE_H_
#define CONTAINER_DEVICE_H_

#include "container_host.h"

class mat_device {
private:
float *pbuf;
int ld;
public:
const int rows;
const int cols;
const int len;
const T_mat_format format;

mat_device(int num_rows, int num_cols, bool init = true, bool pitch = false,
           T_mat_format storage = mat_col_major); mat_device(int num_elements, bool
                                                             init = true);
mat_device(const mat_device &m);
mat_device(const mat_host &m, bool pitch = false);
~mat_device() {
	if (pbuf) HANDLE_ERROR(cudaFree(pbuf));
}

int leading_dim() const {
	return ld;
}

float *data_dev_ptr() {
	return pbuf;
}

const float *data_dev_ptr() const {
	return pbuf;
}

mat_device &operator=(const mat_device &m) throw(illegal_mat_gpu_assignment);
mat_device &operator=(const mat_host &m) throw(illegal_mat_gpu_assignment);
};

class sparse_mat_device {
private:
int *p_ind;
int *p_ptr;
float *p_val;
public:
const int rows;
const int cols;
const int nnz;
T_sparse_mat_format format;

sparse_mat_device(int num_rows, int num_cols, int n_nonzero,
                  T_sparse_mat_format storage_format, bool init = true);
sparse_mat_device(const sparse_mat_device &m);
sparse_mat_device(const sparse_mat_host &m);
~sparse_mat_device();

float *val() {
	return p_val;
}
const float *val() const {
	return p_val;
}

int *ptr() {
	return p_ptr;
}
const int *ptr() const {
	return p_ptr;
}

int *ind() {
	return p_ind;
}
const int *ind() const {
	return p_ind;
}

sparse_mat_device &operator=(const sparse_mat_device &m)
throw(illegal_mat_gpu_assignment);
sparse_mat_device &operator=(const sparse_mat_host &m)
throw(illegal_mat_gpu_assignment);
};



// ------------------------------------------------------------------------
// member functions, etc. class mat_device
// ------------------------------------------------------------------------

// member functions...
mat_device::mat_device(int num_rows, int num_cols, bool init, bool pitch,
                       T_mat_format storage) : pbuf(NULL), rows(num_rows), cols(
	        num_cols),
	len(num_rows * num_cols), format(storage) {

	if (len > 0) {
		if (pitch) {
			size_t width =
			        (format ==
			         mat_row_major) ? cols * sizeof(float) :
			        rows * sizeof(float); size_t height =
			        (format == mat_row_major) ? rows :
			        cols;
			size_t p;
			HANDLE_ERROR(cudaMallocPitch(&pbuf, &p, width, height));
			ld = p / sizeof(float);

			if (init)
				HANDLE_ERROR(cudaMemset2D(pbuf, p, 0, width,
				                          height));
		} else {
			HANDLE_ERROR(cudaMalloc(&pbuf, len * sizeof(float)));
			ld = (format == mat_row_major) ? cols : rows;

			if (init)
				HANDLE_ERROR(cudaMemset(pbuf, 0, len *
				                        sizeof(float)));
		}
	}
};

mat_device::mat_device(int num_elements, bool init) : pbuf(NULL),
	rows(num_elements), cols(1), len(num_elements), format(mat_col_major) {

	if (len > 0) {
		HANDLE_ERROR(cudaMalloc(&pbuf, len * sizeof(float)));
		ld = (format == mat_row_major) ? cols : rows;

		if (init)
			HANDLE_ERROR(cudaMemset(pbuf, 0, len * sizeof(float)));
	}
};

mat_device::mat_device(const mat_device &m) : pbuf(NULL), rows(m.rows),
	cols(m.cols), len(m.len), format(m.format), ld(m.ld) {

	if (len > 0) {
		size_t width =
		        (format == mat_row_major) ? cols * sizeof(float) :
		        rows * sizeof(float); size_t height =
		        (format == mat_row_major) ? rows :
		        cols;
		int total_len =
		        (format == mat_row_major) ? ld * rows : ld * cols;
		HANDLE_ERROR(cudaMalloc(&pbuf, total_len * sizeof(float)));
		HANDLE_ERROR(cudaMemcpy2D(pbuf, ld * sizeof(float), m.pbuf,
		                          m.ld * sizeof(float), width, height,
		                          cudaMemcpyDeviceToDevice));
	}
};

mat_device::mat_device(const mat_host &m,
                       bool pitch) : pbuf(NULL), rows(m.rows),
	cols(m.cols), len(m.len), format(m.format) {

	if (len > 0) {
		if (pitch) {
			size_t width =
			        (format ==
			         mat_row_major) ? cols * sizeof(float) :
			        rows * sizeof(float); size_t height =
			        (format == mat_row_major) ? rows :
			        cols;
			size_t p;

			HANDLE_ERROR(cudaMallocPitch(&pbuf, &p, width, height));
			ld = p / sizeof(float);
			HANDLE_ERROR(cudaMemcpy2D(pbuf, ld * sizeof(float),
			                          m.data(), width, width,
			                          height,
			                          cudaMemcpyHostToDevice));
		} else {
			HANDLE_ERROR(cudaMalloc(&pbuf, len * sizeof(float)));
			ld = (format == mat_row_major) ? cols : rows;
			HANDLE_ERROR(cudaMemcpy(pbuf, m.data(), len *
			                        sizeof(float),
			                        cudaMemcpyHostToDevice));
		}
	}
};

mat_device &mat_device::operator=(const mat_device &m)
throw(illegal_mat_gpu_assignment) {

	int w_m = (m.format == mat_row_major) ? m.cols : m.rows;
	int w = (format == mat_row_major) ? cols : rows;

	if (w_m == m.ld && w == ld && len == m.len) {
		HANDLE_ERROR(cudaMemcpy(pbuf, m.data_dev_ptr(), len *
		                        sizeof(float),
		                        cudaMemcpyDeviceToDevice));
	} else if ((m.rows == rows) && (m.cols ==
	                                cols) && (m.format == format)) {
		size_t width =
		        (format == mat_row_major) ? cols * sizeof(float) :
		        rows * sizeof(float); size_t height =
		        (format == mat_row_major) ? rows :
		        cols;
		HANDLE_ERROR(cudaMemcpy2D(pbuf, ld * sizeof(float), m.pbuf,
		                          m.ld * sizeof(float), width, height,
		                          cudaMemcpyDeviceToDevice));
	} else {
		throw illegal_mat_gpu_assignment(
		              "Illegal assignment of mat_device objects!");
	}
	return *this;
}


mat_device &mat_device::operator=(const mat_host &m)
throw(illegal_mat_gpu_assignment) {

	if ((m.rows == rows) && (m.cols == cols) && (m.format == format)) {
		size_t width =
		        (format == mat_row_major) ? cols * sizeof(float) :
		        rows * sizeof(float); size_t height =
		        (format == mat_row_major) ? rows :
		        cols;
		HANDLE_ERROR(cudaMemcpy2D(pbuf, ld * sizeof(float), m.data(),
		                          width, width,
		                          height, cudaMemcpyHostToDevice));
	} else {
		throw illegal_mat_gpu_assignment(
		              "Illegal assignment mat_device <- mat_host!");
	}
	return *this;
}

// function for conversion mat_device -> mat_host
void mat_gpu_to_host(mat_host &m_host, const mat_device &m)
throw(illegal_mat_gpu_assignment) {

	if ((m.rows == m_host.rows) && (m.cols == m_host.cols) && (m.format ==
	                                                           m_host.format))
	{
		size_t width = (m.format == mat_row_major) ?
		               m.cols * sizeof(float) : m.rows * sizeof(float);
		size_t height = (m.format == mat_row_major) ? m.rows : m.cols;

		HANDLE_ERROR(cudaMemcpy2D(m_host.data(), width,
		                          m.data_dev_ptr(),
		                          m.leading_dim() * sizeof(float),
		                          width, height,
		                          cudaMemcpyDeviceToHost));
	} else {
		throw illegal_mat_gpu_assignment(
		              "Illegal assignment mat_host <- mat_device!");
	}
};

// output operator
std::ostream &operator<<(std::ostream &stream, const mat_device &m) {
	mat_host m_host(m.rows, m.cols, m.format, false, false);

	mat_gpu_to_host(m_host, m);

	stream << m_host;

	return stream;
}

// ------------------------------------------------------------------------
// member functions, etc. class sparse_mat_device
// ------------------------------------------------------------------------

// member functions...
sparse_mat_device::sparse_mat_device(int num_rows, int num_cols, int n_nonzero,
                                     T_sparse_mat_format storage_format,
                                     bool init) : p_val(NULL), p_ptr(NULL),
	p_ind(NULL), rows(num_rows), cols(num_cols), nnz(n_nonzero),
	format(storage_format) {

	if (nnz > 0) {
		int len_ptr = (format == sparse_mat_csc) ? cols + 1 : rows + 1;

		HANDLE_ERROR(cudaMalloc(&p_val, nnz * sizeof(float)));

		if (init)
			HANDLE_ERROR(cudaMemset(p_val, 0, nnz * sizeof(float)));






		HANDLE_ERROR(cudaMalloc(&p_ptr, len_ptr * sizeof(float)));

		if (init)
			HANDLE_ERROR(cudaMemset(p_ptr, 0, len_ptr *
			                        sizeof(float)));
		HANDLE_ERROR(cudaMalloc(&p_ind, nnz * sizeof(float)));

		if (init)
			HANDLE_ERROR(cudaMemset(p_ind, 0, nnz * sizeof(float)));






	}
};

sparse_mat_device::sparse_mat_device(const sparse_mat_device &m) : p_val(NULL),
	p_ptr(NULL), p_ind(NULL), rows(m.rows), cols(m.cols), nnz(m.nnz),
	format(m.format) {

	if (nnz > 0) {
		int len_ptr = (format == sparse_mat_csc) ? cols + 1 : rows + 1;

		HANDLE_ERROR(cudaMalloc(&p_val, nnz * sizeof(float)));
		HANDLE_ERROR(cudaMemcpy(p_val, m.p_val, nnz * sizeof(float),
		                        cudaMemcpyDeviceToDevice));

		HANDLE_ERROR(cudaMalloc(&p_ptr, len_ptr * sizeof(int)));
		HANDLE_ERROR(cudaMemcpy(p_ptr, m.p_ptr, len_ptr * sizeof(int),
		                        cudaMemcpyDeviceToDevice));

		HANDLE_ERROR(cudaMalloc(&p_ind, nnz * sizeof(int)));
		HANDLE_ERROR(cudaMemcpy(p_ind, m.p_ind, nnz * sizeof(int),
		                        cudaMemcpyDeviceToDevice));
	}
};

sparse_mat_device::sparse_mat_device(const sparse_mat_host &m) : p_val(NULL),
	p_ptr(NULL), p_ind(NULL), rows(m.rows), cols(m.cols), nnz(m.nnz),
	format(m.format) {

	if (nnz > 0) {
		int len_ptr = (format == sparse_mat_csc) ? cols + 1 : rows + 1;

		HANDLE_ERROR(cudaMalloc(&p_val, nnz * sizeof(float)));
		HANDLE_ERROR(cudaMemcpy(p_val, m.val(), nnz * sizeof(float),
		                        cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaMalloc(&p_ptr, len_ptr * sizeof(int)));
		HANDLE_ERROR(cudaMemcpy(p_ptr, m.ptr(), len_ptr * sizeof(int),
		                        cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaMalloc(&p_ind, nnz * sizeof(int)));
		HANDLE_ERROR(cudaMemcpy(p_ind, m.ind(), nnz * sizeof(int),
		                        cudaMemcpyHostToDevice));

	}
};

sparse_mat_device::~sparse_mat_device() {
	if (p_val) HANDLE_ERROR(cudaFree(p_val));

	if (p_ptr) HANDLE_ERROR(cudaFree(p_ptr));

	if (p_ind) HANDLE_ERROR(cudaFree(p_ind));
}

sparse_mat_device &sparse_mat_device::operator=(const sparse_mat_device &m)
throw(illegal_mat_gpu_assignment) {

	if (m.nnz == nnz && m.format == format &&
	    m.cols == cols && m.rows == rows) {
		int len_ptr = (format == sparse_mat_csc) ? cols + 1 : rows + 1;

		HANDLE_ERROR(cudaMemcpy(p_val, m.p_val, nnz * sizeof(float),
		                        cudaMemcpyDeviceToDevice));

		HANDLE_ERROR(cudaMemcpy(p_ptr, m.p_ptr, len_ptr * sizeof(int),
		                        cudaMemcpyDeviceToDevice));

		HANDLE_ERROR(cudaMemcpy(p_ind, m.p_ind, nnz * sizeof(int),
		                        cudaMemcpyDeviceToDevice));

	} else {
		throw illegal_mat_gpu_assignment(
		              "Illegal assignment of sparse_mat_device objects!");
	}
	return *this;
}

sparse_mat_device &sparse_mat_device::operator=(const sparse_mat_host &m)
throw(illegal_mat_gpu_assignment) {

	if (m.nnz == nnz && m.format == format &&
	    m.cols == cols && m.rows == rows) {
		int len_ptr = (format == sparse_mat_csc) ? cols + 1 : rows + 1;

		HANDLE_ERROR(cudaMemcpy(p_val, m.val(), nnz * sizeof(float),
		                        cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaMemcpy(p_ptr, m.ptr(), len_ptr * sizeof(int),
		                        cudaMemcpyHostToDevice));

		HANDLE_ERROR(cudaMemcpy(p_ind, m.ind(), nnz * sizeof(int),
		                        cudaMemcpyHostToDevice));

	} else {
		throw illegal_mat_gpu_assignment((
		                                         "Illegal assignment sparse_ mat_device <- sparse_mat_host!"));
	}
	return *this;
}

// function for conversion sparse_mat_device -> sparse_mat_host
void sparse_mat_gpu_to_host(sparse_mat_host &lhs, const sparse_mat_device &rhs)
throw(illegal_mat_gpu_assignment) {

	if (lhs.nnz == rhs.nnz && lhs.format ==
	    rhs.format && lhs.cols == rhs.cols && lhs.rows == rhs.rows) {
		int len_ptr =
		        (lhs.format ==
		         sparse_mat_csc) ? lhs.cols + 1 : lhs.rows + 1;

		HANDLE_ERROR(cudaMemcpy(lhs.val(), rhs.val(), lhs.nnz *
		                        sizeof(float),
		                        cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(lhs.ptr(), rhs.ptr(), len_ptr *
		                        sizeof(int),
		                        cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(lhs.ind(), rhs.ind(), lhs.nnz *
		                        sizeof(int),
		                        cudaMemcpyDeviceToHost));
	} else {
		throw illegal_mat_gpu_assignment(
		              "Illegal assignment sparse_ mat_host <- sparse_mat_device!");
	}
}

// output operator
std::ostream &operator<<(std::ostream &stream, const sparse_mat_device &m) {
	sparse_mat_host tmp(m.rows, m.cols, m.nnz, m.format);

	sparse_mat_gpu_to_host(tmp, m);

	stream << tmp;

	return stream;
}

#endif /* CONTAINER_DEVICE_H_ */
