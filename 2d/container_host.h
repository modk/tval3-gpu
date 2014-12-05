#ifndef CONTAINER_HOST_H_
#define CONTAINER_HOST_H_

#include <cuda_runtime.h>
#include "handle_error.h"
#include <cstring>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

// enumerations...
enum T_alloc {allocated_in_constructor, preallocated_buffer};
enum T_mat_format {mat_col_major, mat_row_major};
enum T_sparse_mat_format {sparse_mat_csc, sparse_mat_csr};

// exceptions...

class illegal_mat_gpu_assignment : public std::runtime_error {
	public:
		illegal_mat_gpu_assignment(const std::string& message) : 
			std::runtime_error(message) { };
};

//---------------------------------------
// class mat_host...
//---------------------------------------

class mat_host {
	private:
		float *pbuf;
		T_alloc alloc_info;
	public:
		const int rows;
		const int cols;
		const int len;
		const T_mat_format format;
		const bool pagelocked;
		const unsigned int alloc_flags;

		mat_host(int num_rows, int num_cols, T_mat_format storage = mat_col_major,
				bool init = true, bool mem_pagelocked = false,
				unsigned int flags = cudaHostAllocDefault);
		mat_host(int num_elements, bool init = true, bool mem_pagelocked = false,
				unsigned int flags = cudaHostAllocDefault);
		mat_host(int num_rows, int num_cols, float *buffer,
				T_mat_format storage = mat_col_major,
				bool mem_pagelocked = false, unsigned int flags =
				cudaHostAllocDefault);
		mat_host(int num_elements, float *buffer, bool mem_pagelocked = false,
				unsigned int flags = cudaHostAllocDefault);
		mat_host(const mat_host &m);
		~mat_host();

		float *data() {
			return pbuf;
		}
		const float *data() const {
			return pbuf;
		}

		float &operator[](int i) {
			return pbuf[i];
		}
		const float &operator[](int i) const {
			return pbuf[i];
		}

		mat_host &operator=(const mat_host &m) throw(illegal_mat_gpu_assignment);

		void randomFill(float scl);
};


//---------------------------------------
// class sparse_mat_host...
//---------------------------------------

class sparse_mat_host {
private:
int *p_ptr;
int *p_ind;
float *p_val;
T_alloc alloc_info;
public:
const int rows;
const int cols;
const int nnz;
const T_sparse_mat_format format;
const bool pagelocked;
const unsigned int alloc_flags;

sparse_mat_host(int num_rows, int num_cols, int n_nonzero,
                T_sparse_mat_format storage_format = sparse_mat_csc,
                bool init = true,
                bool mem_pagelocked = false,
                unsigned int flags = cudaHostAllocDefault);
sparse_mat_host(int num_rows, int num_cols, int n_nonzero, float *buff_val,
                int *buff_ptr, int *buff_ind,
                T_sparse_mat_format storage_format = sparse_mat_csc,
                bool mem_pagelocked = false,
                unsigned int flags = cudaHostAllocDefault);
sparse_mat_host(const sparse_mat_host &m);
~sparse_mat_host();

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

sparse_mat_host &operator=(const sparse_mat_host &m) throw(
        illegal_mat_gpu_assignment);
};

// ------------------------------------------------------------------------
// member functions, etc. class mat_host
// ------------------------------------------------------------------------

mat_host::mat_host(int num_rows, int num_cols, T_mat_format storage, bool init,
                   bool mem_pagelocked, unsigned int flags) : pbuf(NULL),
	rows(num_rows), cols(num_cols), len(num_rows * num_cols), alloc_info(
	        allocated_in_constructor), format(storage),
	pagelocked(mem_pagelocked), alloc_flags(flags) {
	if (len > 0) {
		if (pagelocked)
			HANDLE_ERROR(cudaHostAlloc(&pbuf, len * sizeof(float),
			                           alloc_flags));
		else
			pbuf = new float[len];

		if (init)
			memset(pbuf, 0, len * sizeof(float));
	}
};


mat_host::mat_host(int num_elements, bool init, bool mem_pagelocked,
                   unsigned int flags) : pbuf(NULL), rows(num_elements),
	cols(1), len(num_elements), alloc_info(allocated_in_constructor),
	format(mat_col_major), pagelocked(mem_pagelocked),
	alloc_flags(flags) {
	if (len > 0) {
		if (pagelocked)
			HANDLE_ERROR(cudaHostAlloc(&pbuf, len * sizeof(float),
			                           alloc_flags));
		else
			pbuf = new float[len];

		if (init)
			memset(pbuf, 0, len * sizeof(float));
	}
};


mat_host::mat_host(int num_rows, int num_cols, float *buffer,
                   T_mat_format storage, bool mem_pagelocked,
                   unsigned int flags) :
	pbuf(buffer), rows(num_rows), cols(num_cols), len(num_rows * num_cols),
	alloc_info(preallocated_buffer), format(storage),
	pagelocked(mem_pagelocked), alloc_flags(flags) {
};


mat_host::mat_host(int num_elements, float *buffer, bool mem_pagelocked,
                   unsigned int flags) : pbuf(buffer), rows(num_elements),
	cols(1), len(num_elements), alloc_info(preallocated_buffer), format(
	        mat_row_major),
	pagelocked(mem_pagelocked), alloc_flags(flags) {
};


mat_host::mat_host(const mat_host &m) : pbuf(NULL), rows(m.rows), cols(m.cols),
	len(m.len),
	alloc_info(allocated_in_constructor), format(m.format), pagelocked(
	        m.pagelocked), alloc_flags(m.alloc_flags) {
	if (len > 0) {
		if (pagelocked)
			HANDLE_ERROR(cudaHostAlloc(&pbuf, len * sizeof(float),
			                           alloc_flags));
		else
			pbuf = new float[len];
		memcpy(pbuf, m.pbuf, len * sizeof(float));
	}
};


mat_host::~mat_host() {
	if (pbuf && (alloc_info == allocated_in_constructor)) {
		if (pagelocked) {
			HANDLE_ERROR(cudaFreeHost(pbuf));
		} else {
			delete[] pbuf;
		}
	}
}


mat_host &mat_host::operator=(const mat_host &m) throw(
        illegal_mat_gpu_assignment) {
	if (m.len == len)
		memcpy(pbuf, m.pbuf, len * sizeof(float));
	else
		throw illegal_mat_gpu_assignment((
		                                         "Illegale Zuweisung von mat_host-Objekten!"));






	return *this;
}


void mat_host::randomFill(float scl) {
	for (int i = 0; i < len; i++)
		pbuf[i] = ((float)rand() / RAND_MAX - 0.5) / scl;
}


std::ostream &operator<<(std::ostream &stream, const mat_host &m) {
	for (int i = 0; i < m.rows; i++) {
		for (int j = 0; j < m.cols; j++) {
			if (m.format == mat_row_major)
				stream << m[i * m.cols + j] << " ";
			else
				stream << m[i + j * m.rows] << " ";
		}
		stream << "\n";
	}
	return stream;
}


// ------------------------------------------------------------------------
// member functions, etc. class sparse_mat_host
// ------------------------------------------------------------------------


sparse_mat_host::sparse_mat_host(int num_rows, int num_cols, int n_nonzero,
                                 T_sparse_mat_format storage_format, bool init,
                                 bool mem_pagelocked, unsigned int flags) :
	p_val(NULL), p_ptr(NULL), p_ind(NULL), rows(num_rows), cols(num_cols),
	nnz(n_nonzero), format(storage_format),
	alloc_info(allocated_in_constructor), pagelocked(mem_pagelocked),
	alloc_flags(flags) {
	if (nnz > 0) {
		int len_ptr = (format == sparse_mat_csc) ? cols + 1 : rows + 1;

		if (pagelocked) {
			HANDLE_ERROR(cudaHostAlloc(&p_val, nnz * sizeof(float),
			                           alloc_flags));
			HANDLE_ERROR(cudaHostAlloc(&p_ptr, len_ptr *
			                           sizeof(int), alloc_flags));
			HANDLE_ERROR(cudaHostAlloc(&p_ind, nnz * sizeof(int),
			                           alloc_flags));
		} else {
			p_val = new float[nnz];
			p_ptr = new int[len_ptr];
			p_ind = new int[nnz];
		}

		if (init) {
			memset(p_val, 0, nnz * sizeof(float));
			memset(p_ptr, 0, (len_ptr) * sizeof(int));
			memset(p_ind, 0, nnz * sizeof(int));
		}
	}
};


sparse_mat_host::sparse_mat_host(int num_rows, int num_cols, int n_nonzero,
                                 float *buff_val, int *buff_ptr, int *buff_ind,
                                 T_sparse_mat_format storage_format,
                                 bool mem_pagelocked,
                                 unsigned int flags) : p_val(buff_val), p_ptr(
	        buff_ptr),
	p_ind(buff_ind), rows(num_rows), cols(num_cols), nnz(n_nonzero), format(
	        storage_format),
	alloc_info(preallocated_buffer), pagelocked(mem_pagelocked),
	alloc_flags(flags) {
}



sparse_mat_host::sparse_mat_host(const sparse_mat_host &m) : p_val(NULL), p_ptr(
	        NULL), p_ind(NULL), rows(m.rows),
	cols(m.cols), nnz(m.nnz), format(m.format), alloc_info(
	        allocated_in_constructor),
	pagelocked(m.pagelocked), alloc_flags(m.alloc_flags) {
	if (nnz > 0) {
		int len_ptr = (format == sparse_mat_csc) ? cols + 1 : rows + 1;

		if (pagelocked) {
			HANDLE_ERROR(cudaHostAlloc(&p_val, nnz * sizeof(float),
			                           alloc_flags));
			HANDLE_ERROR(cudaHostAlloc(&p_ptr, len_ptr *
			                           sizeof(int), alloc_flags));
			HANDLE_ERROR(cudaHostAlloc(&p_ind, nnz * sizeof(int),
			                           alloc_flags));
		} else {
			p_val = new float[nnz];
			p_ptr = new int[len_ptr];
			p_ind = new int[nnz];
		}
		memcpy(p_val, m.p_val, nnz * sizeof(float));
		memcpy(p_ptr, m.p_ptr, (len_ptr) * sizeof(int));
		memcpy(p_ind, m.p_ind, nnz * sizeof(int));
	}
};


sparse_mat_host::~sparse_mat_host() {
	if (alloc_info == allocated_in_constructor) {
		if (pagelocked) {
			if (p_val) HANDLE_ERROR(cudaFreeHost(p_val));

			if (p_ptr) HANDLE_ERROR(cudaFreeHost(p_ptr));

			if (p_ind) HANDLE_ERROR(cudaFreeHost(p_ind));
		} else {
			if (p_val) delete[] p_val;

			if (p_ptr) delete[] p_ptr;

			if (p_ind) delete[] p_ind;
		}
	}
}


sparse_mat_host &sparse_mat_host::operator=(const sparse_mat_host &m) throw(
        illegal_mat_gpu_assignment){

	if (m.nnz == nnz && m.rows == rows && m.cols == cols && m.format ==
	    format) {
		int len_ptr = (format == sparse_mat_csc) ? cols + 1 : rows + 1;

		memcpy(p_val, m.p_val, nnz * sizeof(float));
		memcpy(p_ptr, m.p_ptr, (len_ptr) * sizeof(int));
		memcpy(p_ind, m.p_ind, nnz * sizeof(int));
	} else {
		throw illegal_mat_gpu_assignment((
		                                         "Illegale Zuweisung von sparse_mat_host-Objekten!"));
	}

	return *this;
}

// output operator ...

std::ostream &operator<<(std::ostream &stream, const sparse_mat_host &m) {
	mat_host tmp(m.rows, m.cols);

	if (m.format == sparse_mat_csc) {
		for (int c = 0; c < m.cols; c++)
			for (int i = m.ptr()[c]; i < m.ptr()[c + 1]; i++)
				tmp[c * m.rows + m.ind()[i]] = m.val()[i];
	} else {
		for (int r = 0; r < m.rows; r++)
			for (int i = m.ptr()[r]; i < m.ptr()[r + 1]; i++)
				tmp[m.ind()[i] * m.rows + r] = m.val()[i];
	}
	stream << tmp;
	return stream;
}

#endif /* CONTAINER_HOST_H_ */
