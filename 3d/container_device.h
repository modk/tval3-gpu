#ifndef CONTAINER_DEVICE_H_
#define CONTAINER_DEVICE_H_

#include "container_host.h"

//-----------------------------------------------------------------------------
// class mat_device: definition
//-----------------------------------------------------------------------------
template <class Type>
class mat_device {
private:
Type *pbuf;
int ld;
public:
const int dim_x;
const int dim_y;
const int dim_z;
const int len;
const T_mat_format format;

mat_device(int l_y, int l_x, int l_z, bool init = true, bool pitch = false,
           T_mat_format storage = mat_col_major);
mat_device(int num_elements, bool init = true);
mat_device(const mat_device &m);
mat_device(const mat_host<Type> &m, bool pitch = false);
~mat_device() {
	if (pbuf) HANDLE_ERROR(cudaFree(pbuf));
}

int leading_dim() const {
	return ld;
}

Type *data_dev_ptr() {
	return pbuf;
}
const Type *data_dev_ptr() const {
	return pbuf;
}

mat_device &operator=(const mat_device<Type> &m) throw(
        illegal_mat_gpu_assignment);
mat_device &operator=(const mat_host<Type> &m) throw(illegal_mat_gpu_assignment);
};
//-----------------------------------------------------------------------------
// class sparse_mat_device: definition
//-----------------------------------------------------------------------------
template <class Type>
class sparse_mat_device {
	private:
		int *p_ind;
		int *p_ptr;
		Type *p_val;
	public:
		const int dim_y;
		const int dim_x;
		const int nnz;
		T_sparse_mat_format format;

		sparse_mat_device(int num_dim_y, int num_dim_x, int n_nonzero,
				T_sparse_mat_format storage_format, bool init =
				true) throw(illegal_mat_gpu_assignment);
		sparse_mat_device(int dim_y, int dim_x, int nnz, const Type *buff_val,
				const int *buff_ptr, const int *buff_ind,
				T_sparse_mat_format storage_format) throw(
					illegal_mat_gpu_assignment);
		sparse_mat_device(const sparse_mat_device &m);
		sparse_mat_device(const sparse_mat_host<Type> &m,
				T_sparse_mat_format storage_format) throw(
					illegal_mat_gpu_assignment);
		~sparse_mat_device();

		Type *val() {
			return p_val;
		}
		const Type *val() const {
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

		sparse_mat_device &operator=(const sparse_mat_device<Type> &m) throw(
				illegal_mat_gpu_assignment);
		sparse_mat_device &operator=(const sparse_mat_host<Type> &m) throw(
				illegal_mat_gpu_assignment);
};

//-----------------------------------------------------------------------------
// class geometry_device: definition
//-----------------------------------------------------------------------------
class geometry_device {
	private:
		int *x_emitters;
		int *y_emitters;
		int *z_emitters;
		int *x_receivers;
		int *y_receivers;
		int *z_receivers;
		unsigned int *use_path;

	public:
		int *x_em_dev_ptr() {
			return x_emitters;
		}
		const int *x_em_dev_ptr() const {
			return x_emitters;
		}
		int *y_em_dev_ptr() {
			return y_emitters;
		}
		const int *y_em_dev_ptr() const {
			return y_emitters;
		}
		int *z_em_dev_ptr() {
			return z_emitters;
		}
		const int *z_em_dev_ptr() const {
			return z_emitters;
		}
		int *x_re_dev_ptr() {
			return x_receivers;
		}
		const int *x_re_dev_ptr() const {
			return x_receivers;
		}
		int *y_re_dev_ptr() {
			return y_receivers;
		}
		const int *y_re_dev_ptr() const {
			return y_receivers;
		}
		int *z_re_dev_ptr() {
			return z_receivers;
		}
		const int *z_re_dev_ptr() const {
			return z_receivers;
		}
		unsigned int *use_path_dev_ptr() {
			return use_path;
		}
		const unsigned int *use_path_dev_ptr() const {
			return use_path;
		}

		const unsigned int numMPs;
		const unsigned int numPaths;
		const unsigned int numSignals;
		const int num_emitters;
		const int num_receivers;
		const int rv_x;
		const int rv_y;
		const int rv_z;
		const float scale_factor;
		const int ld;

		geometry_device(const geometry_host &geom_host, unsigned int *dev_use_path,
				unsigned int dev_num_paths, unsigned int dev_num_signals);
		geometry_device(const geometry_device &geom_dev);
		~geometry_device();
};

//-----------------------------------------------------------------------------
// class mat_device: member functions
//-----------------------------------------------------------------------------
#if (1)
template<class Type>
mat_device<Type>::mat_device(int l_y, int l_x, int l_z, bool init, bool pitch,
                             T_mat_format storage) : pbuf(NULL), dim_y(l_y),
	dim_z(l_z), dim_x(l_x), len(l_y * l_x * l_z), format(storage) {

	if (len > 0) {
		if (pitch) {
			size_t width =
			        (format ==
			         mat_row_major) ? dim_x * sizeof(Type) : dim_y *
			        sizeof(Type);
			size_t height =
			        (format == mat_row_major) ? dim_y : dim_x;
			size_t p;
			HANDLE_ERROR(cudaMallocPitch(&pbuf, &p, width, height *
			                             dim_z));
			ld = p / sizeof(Type);

			if (init)
				HANDLE_ERROR(cudaMemset2D(pbuf, p, 0, width,
				                          height * dim_z));
		} else {
			HANDLE_ERROR(cudaMalloc(&pbuf, len * sizeof(Type)));
			ld = (format == mat_row_major) ? dim_x : dim_y;

			if (init)
				HANDLE_ERROR(cudaMemset(pbuf, 0, len *
				                        sizeof(Type)));
		}
	}
};

template<class Type>
mat_device<Type>::mat_device(int num_elements, bool init) : pbuf(NULL), dim_y(
	        num_elements), dim_x(1), dim_z(1), len(num_elements),
	format(mat_col_major) {
	if (len > 0) {
		HANDLE_ERROR(cudaMalloc(&pbuf, len * sizeof(Type)));
		ld = (format == mat_row_major) ? dim_x : dim_y;

		if (init)
			HANDLE_ERROR(cudaMemset(pbuf, 0, len * sizeof(Type)));
	}
};

template<class Type>
mat_device<Type>::mat_device(const mat_device &m) : pbuf(NULL), dim_y(m.dim_y),
	dim_x(m.dim_x), dim_z(m.dim_z), len(m.len),
	format(m.format), ld(m.ld) {
	if (len > 0) {
		size_t width =
		        (format ==
		         mat_row_major) ? dim_x * sizeof(Type) : dim_y *
		        sizeof(Type);
		size_t height = (format == mat_row_major) ? dim_y : dim_x;
		int total_len =
		        (format ==
		         mat_row_major) ? ld * dim_y * dim_z : ld * dim_x *
		        dim_z;
		HANDLE_ERROR(cudaMalloc(&pbuf, total_len * sizeof(Type)));
		HANDLE_ERROR(cudaMemcpy2D(pbuf, ld * sizeof(Type), m.pbuf,
		                          m.ld * sizeof(Type), width, height *
		                          dim_z,
		                          cudaMemcpyDeviceToDevice));
	}
};

template<class Type>
mat_device<Type>::mat_device(const mat_host<Type> &m,
                             bool pitch) : pbuf(NULL), dim_y(m.dim_y), dim_x(
	        m.dim_x), dim_z(m.dim_z), len( m.len), format(m.format) {

	if (len > 0) {

		if (pitch) {

			size_t width = (format == mat_row_major) ? 
				dim_x * sizeof(Type) : dim_y * sizeof(Type);
			size_t height = (format == mat_row_major) ? dim_y : dim_x;
			size_t p;

			HANDLE_ERROR(cudaMallocPitch(&pbuf, &p, width, height *
			                             dim_z));
			ld = p / sizeof(Type);
			HANDLE_ERROR(cudaMemcpy2D(pbuf, ld * sizeof(Type),
			                          m.data(), width, width,
			                          height * dim_z,
			                          cudaMemcpyHostToDevice));
		} else {

			HANDLE_ERROR(cudaMalloc(&pbuf, len * sizeof(Type)));
			ld = (format == mat_row_major) ? dim_x : dim_y;
			HANDLE_ERROR(cudaMemcpy(pbuf, m.data(), len * sizeof(Type),
			                        cudaMemcpyHostToDevice));
		}
	}
};

template<class Type>
mat_device<Type> &mat_device<Type>::operator=(const mat_device<Type> &m) throw(
        illegal_mat_gpu_assignment) {

	int w_m = (m.format == mat_row_major) ? m.dim_x : m.dim_y;
	int w = (format == mat_row_major) ? dim_x : dim_y;

	if (w_m == m.ld && w == ld && len == m.len) {
		HANDLE_ERROR(cudaMemcpy(pbuf, m.data_dev_ptr(), len *
		                        sizeof(Type),
		                        cudaMemcpyDeviceToDevice));
	} else if ((m.dim_y == dim_y) && (m.dim_x == dim_x) &&
	           (m.dim_z == dim_z) && (m.format == format)) {
		size_t width = (format == mat_row_major) ? dim_x * sizeof(Type) : 
			dim_y * sizeof(Type);
		size_t height = (format == mat_row_major) ? dim_y : dim_x;
		HANDLE_ERROR(cudaMemcpy2D(pbuf, ld * sizeof(Type), m.pbuf,
		                          m.ld * sizeof(Type), width, height *
		                          dim_z, cudaMemcpyDeviceToDevice));
	} else {
		throw illegal_mat_gpu_assignment(
		              "Illegal Assignment of mat_device objects");
	}
	return *this;
}


template<class Type>
mat_device<Type> &mat_device<Type>::operator=(const mat_host<Type> &m) throw(
        illegal_mat_gpu_assignment) {
	if ((m.dim_y == dim_y) && (m.dim_x == dim_x) && (m.dim_z == dim_z) &&
	    (m.format == format)) {
		size_t width =
		        (format ==
		         mat_row_major) ? dim_x * sizeof(Type) : dim_y *
		        sizeof(Type);
		size_t height = (format == mat_row_major) ? dim_y : dim_x;
		HANDLE_ERROR(cudaMemcpy2D(pbuf, ld * sizeof(Type), m.data(),
		                          width, width, height * dim_z,
		                          cudaMemcpyHostToDevice));
	} else {
		throw illegal_mat_gpu_assignment(
		              "Illegal assignment mat_device <- mat_host");
	}
	return *this;
}

// function for conversion mat_device -> mat_host
template<class Type>
void mat_gpu_to_host(mat_host<Type> &m_host, const mat_device<Type> &m) 
	throw(illegal_mat_gpu_assignment) {

	if ((m.dim_y == m_host.dim_y) && (m.dim_x == m_host.dim_x) &&
	    (m.dim_z == m_host.dim_z) && (m.format == m_host.format)) {

		size_t width = (m.format == mat_row_major) ? m.dim_x * sizeof(Type) : 
			m.dim_y * sizeof(Type);
		size_t height = (m.format == mat_row_major) ? m.dim_y : m.dim_x;
		HANDLE_ERROR(cudaMemcpy2D(m_host.data(), width,
		                          m.data_dev_ptr(), m.leading_dim() *
		                          sizeof(Type), width,
		                          height * m.dim_z, cudaMemcpyDeviceToHost));
	} else {
		throw illegal_mat_gpu_assignment(
		              "Illegal assignment mat_host <- mat_device");
	}
};

// output operator
template<class Type>
std::ostream &operator<<(std::ostream &stream, const mat_device<Type> &m) {

	mat_host<Type> m_host(m.dim_y, m.dim_x, m.dim_z, m.format, false, false);

	mat_gpu_to_host(m_host, m);

	stream << m_host;

	return stream;
}
#endif

//-----------------------------------------------------------------------------
// class sparse_mat_device: member functions
//-----------------------------------------------------------------------------
#if (1)
template<class Type>
sparse_mat_device<Type>::sparse_mat_device(int num_dim_y, int num_dim_x,
                                           int n_nonzero,
                                           T_sparse_mat_format storage_format,
                                           bool init) throw(
        illegal_mat_gpu_assignment) :
	p_val(NULL), p_ptr(NULL), p_ind(NULL), dim_y(num_dim_y),
	dim_x(num_dim_x), nnz(n_nonzero), format(storage_format) {
	if (format == sparse_mat_both) {

		throw illegal_mat_gpu_assignment(
				"Type sparse_mat_device only supports CSR or CSC format, not both. " \ 
				"Use type sparse_mm instead!");

	} else {

		if (nnz > 0) {

			int len_ptr = (format == sparse_mat_csc) ? dim_x + 1 : dim_y + 1;

			HANDLE_ERROR(cudaMalloc(&p_val, nnz * sizeof(Type)));

			if (init) {
				HANDLE_ERROR(cudaMemset(p_val, 0, nnz * sizeof(Type)));
			}
			HANDLE_ERROR(cudaMalloc(&p_ptr, len_ptr * sizeof(Type)));

			if (init) {
				HANDLE_ERROR(cudaMemset(p_ptr, 0, len_ptr * sizeof(Type)));
			}
			HANDLE_ERROR(cudaMalloc(&p_ind, nnz * sizeof(Type)));

			if (init) {
				HANDLE_ERROR(cudaMemset(p_ind, 0, nnz * sizeof(Type)));
			}
		}
	}
};

template<class Type>
sparse_mat_device<Type>::sparse_mat_device(const sparse_mat_device &m) : 
	p_val(NULL), p_ptr(NULL), p_ind(NULL), dim_y(m.dim_y),
	dim_x(m.dim_x), nnz(m.nnz), format(m.format) {

	if (nnz > 0) {

		int len_ptr = (format == sparse_mat_csc) ? dim_x + 1 : dim_y + 1;

		HANDLE_ERROR(cudaMalloc(&p_val, nnz * sizeof(Type)));
		HANDLE_ERROR(cudaMemcpy(p_val, m.p_val, nnz * sizeof(Type),
		                        cudaMemcpyDeviceToDevice));
		HANDLE_ERROR(cudaMalloc(&p_ptr, len_ptr * sizeof(int)));
		HANDLE_ERROR(cudaMemcpy(p_ptr, m.p_ptr, len_ptr * sizeof(int),
		                        cudaMemcpyDeviceToDevice));
		HANDLE_ERROR(cudaMalloc(&p_ind, nnz * sizeof(int)));
		HANDLE_ERROR(cudaMemcpy(p_ind, m.p_ind, nnz * sizeof(int),
		                        cudaMemcpyDeviceToDevice));
	}
};

template<class Type>
sparse_mat_device<Type>::sparse_mat_device(const sparse_mat_host<Type> &m,
		T_sparse_mat_format storage_format) throw(illegal_mat_gpu_assignment) :
	p_val(NULL), p_ptr(NULL), p_ind(NULL), dim_y(m.dim_y), dim_x(m.dim_x),
	nnz(m.nnz), format(storage_format) {

	if ( storage_format == sparse_mat_both ) {

		throw illegal_mat_gpu_assignment(
		              "Type sparse_mat_device only supports CSR or CSC format, not both. Use type sparse_mm instead!");

	} else if ( (storage_format !=
	             m.format) & (m.format != sparse_mat_both) ) {

		throw illegal_mat_gpu_assignment(
		              "Chosen format not available in sparse_mat_host!");

	} else {

		int len_ptr;

		const Type *val;
		const int *ptr, *ind;

		if (nnz > 0) {

			if (format == sparse_mat_csc) {
				len_ptr = dim_x + 1;
				val = m.csc_val();
				ptr = m.csc_ptr();
				ind = m.csc_ind();
			} else {
				len_ptr = dim_y + 1;
				val = m.csr_val();
				ptr = m.csr_ptr();
				ind = m.csr_ind();
			}

			HANDLE_ERROR(cudaMalloc(&p_val, nnz * sizeof(Type)));
			HANDLE_ERROR(cudaMemcpy(p_val, val, nnz * sizeof(Type),
			                        cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMalloc(&p_ptr, len_ptr * sizeof(int)));
			HANDLE_ERROR(cudaMemcpy(p_ptr, ptr, len_ptr * sizeof(int),
			                        cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMalloc(&p_ind, nnz * sizeof(int)));
			HANDLE_ERROR(cudaMemcpy(p_ind, ind, nnz * sizeof(int),
			                        cudaMemcpyHostToDevice));
		}
	}
};

template<class Type>
sparse_mat_device<Type>::sparse_mat_device(int dim_y, int dim_x, int nnz,
		const Type *val, const int *ptr,
		const int *ind,
		T_sparse_mat_format storage_format) throw(illegal_mat_gpu_assignment) :
	p_val(NULL), p_ptr(NULL), p_ind(NULL), dim_y(dim_y), dim_x(dim_x), nnz(
			nnz), format(storage_format) {

	if ( storage_format == sparse_mat_both ) {

		throw illegal_mat_gpu_assignment(
		              "Type sparse_mat_device only supports CSR or CSC format, " \
									"not both. Use type sparse_mm instead!");

	} else {

		int len_ptr;

		if (nnz > 0) {

			if (format == sparse_mat_csc) {
				len_ptr = dim_x + 1;
			} else {
				len_ptr = dim_y + 1;
			}

			HANDLE_ERROR(cudaMalloc((void**)&p_val, nnz * sizeof(Type)));
			HANDLE_ERROR(cudaMemcpy(p_val, val, nnz * sizeof(Type),
			                        cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMalloc((void**)&p_ptr, len_ptr * sizeof(int)));
			HANDLE_ERROR(cudaMemcpy(p_ptr, ptr, len_ptr * sizeof(int),
			                        cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMalloc((void**)&p_ind, nnz * sizeof(int)));
			HANDLE_ERROR(cudaMemcpy(p_ind, ind, nnz * sizeof(int),
			                        cudaMemcpyHostToDevice));
		}
	}
};

template<class Type>
sparse_mat_device<Type>::~sparse_mat_device() {
	if (p_val) HANDLE_ERROR(cudaFree(p_val));

	if (p_ptr) HANDLE_ERROR(cudaFree(p_ptr));

	if (p_ind) HANDLE_ERROR(cudaFree(p_ind));
}

template<class Type>
sparse_mat_device<Type> &sparse_mat_device<Type>::operator=(
		const sparse_mat_device<Type> &m) throw(illegal_mat_gpu_assignment) {

	if (m.nnz == nnz && m.format == format && m.dim_x == dim_x && m.dim_y ==
	    dim_y) {

		int len_ptr = (format == sparse_mat_csc) ? dim_x + 1 : dim_y + 1;

		HANDLE_ERROR(cudaMemcpy(p_val, m.p_val, nnz * sizeof(Type),
		                        cudaMemcpyDeviceToDevice));
		HANDLE_ERROR(cudaMemcpy(p_ptr, m.p_ptr, len_ptr * sizeof(int),
		                        cudaMemcpyDeviceToDevice));
		HANDLE_ERROR(cudaMemcpy(p_ind, m.p_ind, nnz * sizeof(int),
		                        cudaMemcpyDeviceToDevice));
	} else {
		throw illegal_mat_gpu_assignment(
		              "Illegal assignment of sparse_mat_device objects");
	}
	return *this;
}

template<class Type>
sparse_mat_device<Type> &sparse_mat_device<Type>::operator=(
        const sparse_mat_host<Type> &m) throw(illegal_mat_gpu_assignment) {

	if (m.nnz == nnz && m.format == format && m.dim_x == dim_x && m.dim_y ==
	    dim_y) {

		int len_ptr;
		const Type *val;
		const int *ptr, *ind;

		if (format == sparse_mat_csc) {
			len_ptr = dim_x + 1;
			val = m.csc_val();
			ptr = m.csc_ptr();
			ind = m.csc_ind();
		} else {
			len_ptr = dim_y + 1;
			val = m.csr_val();
			ptr = m.csr_ptr();
			ind = m.csr_ind();
		}

		HANDLE_ERROR(cudaMemcpy(p_val, val, nnz * sizeof(Type),
		                        cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(p_ptr, ptr, len_ptr * sizeof(int),
		                        cudaMemcpyHostToDevice));
		HANDLE_ERROR(cudaMemcpy(p_ind, ind, nnz * sizeof(int),
		                        cudaMemcpyHostToDevice));

	} else {
		throw illegal_mat_gpu_assignment(
				"Illegal assignment sparse_ mat_device <- sparse_mat_host");
	}
	return *this;
}

// function for conversion sparse_mat_device -> sparse_mat_host
template<class Type>
void sparse_mat_gpu_to_host(sparse_mat_host<Type> &lhs,
                            const sparse_mat_device<Type> &rhs) 
	throw(illegal_mat_gpu_assignment) {

	if (lhs.nnz == rhs.nnz && lhs.format == rhs.format && lhs.dim_x ==
	    rhs.dim_x && lhs.dim_y == rhs.dim_y) {

		int len_ptr;
		Type *val;
		int *ptr, *ind;

		if (rhs.format == sparse_mat_csc) {
			len_ptr = lhs.dim_x + 1;
			val = lhs.csc_val();
			ptr = lhs.csc_ptr();
			ind = lhs.csc_ind();
		} else {
			len_ptr = lhs.dim_y + 1;
			val = lhs.csr_val();
			ptr = lhs.csr_ptr();
			ind = lhs.csr_ind();
		}

		HANDLE_ERROR(cudaMemcpy(val, rhs.val(), lhs.nnz * sizeof(Type),
		                        cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(ptr, rhs.ptr(), len_ptr * sizeof(int),
		                        cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(ind, rhs.ind(), lhs.nnz * sizeof(int),
		                        cudaMemcpyDeviceToHost));
	} else {
		throw illegal_mat_gpu_assignment(
				"Illegal assignment sparse_ mat_host <- sparse_mat_device");
	}
}

template<class Type>
void sparse_mat_gpu_to_host(sparse_mat_host<Type> &lhs,
		const sparse_mat_device<Type> *rhs) throw(illegal_mat_gpu_assignment) {

	if (lhs.nnz == rhs->nnz && lhs.format == rhs->format && lhs.dim_x ==
	    rhs->dim_x && lhs.dim_y == rhs->dim_y) {

		int len_ptr;
		Type *val;
		int *ptr, *ind;

		if (rhs->format == sparse_mat_csc) {
			len_ptr = lhs.dim_x + 1;
			val = lhs.csc_val();
			ptr = lhs.csc_ptr();
			ind = lhs.csc_ind();
		} else {
			len_ptr = lhs.dim_y + 1;
			val = lhs.csr_val();
			ptr = lhs.csr_ptr();
			ind = lhs.csr_ind();
		}

		HANDLE_ERROR(cudaMemcpy(val, rhs->val(), lhs.nnz * sizeof(Type),
		                        cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(ptr, rhs->ptr(), len_ptr * sizeof(int),
		                        cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(ind, rhs->ind(), lhs.nnz * sizeof(int),
		                        cudaMemcpyDeviceToHost));
	} else {
		throw illegal_mat_gpu_assignment(
				"Illegal assignment sparse_ mat_host <- sparse_mat_device");
	}
}

// output operator
template<class Type>
std::ostream &operator<<(std::ostream &stream,
                         const sparse_mat_device<Type> &m) {
	sparse_mat_host<Type> tmp(m.dim_y, m.dim_x, m.nnz, m.format);

	sparse_mat_gpu_to_host(tmp, m);

	stream << tmp;

	return stream;
}

// output operator
template<class Type>
std::ostream &operator<<(std::ostream &stream,
                         const sparse_mat_device<Type> *m) {
	sparse_mat_host<Type> tmp(m->dim_y, m->dim_x, m->nnz, m->format);

	sparse_mat_gpu_to_host(tmp, m);

	stream << tmp;

	return stream;
}
#endif

//-----------------------------------------------------------------------------
// class geometry_device: member functions
//-----------------------------------------------------------------------------
#if (1)
geometry_device::geometry_device(const geometry_host &geom_host,
                                 unsigned int *host_use_path,
                                 unsigned int host_num_paths,
                                 unsigned int host_num_signals) :
	num_emitters(geom_host.num_emitters), num_receivers(geom_host.num_receivers),
	rv_x(geom_host.rv_x), rv_y(geom_host.rv_y), rv_z(geom_host.rv_z),
	scale_factor(geom_host.scale_factor),
	x_emitters(NULL), y_emitters(NULL), z_emitters(NULL), x_receivers(NULL),
	y_receivers(NULL), z_receivers(NULL),
	numMPs(geom_host.numMPs), use_path(NULL), ld(geom_host.ld), 
	numPaths(host_num_paths), numSignals(host_num_signals) {

	if (numMPs > 0) {

		if (numPaths > 0 ) {
			HANDLE_ERROR(cudaMalloc((void**)&use_path, 2 *
			                        numPaths * sizeof(unsigned int)));

			HANDLE_ERROR(cudaMemcpy(use_path, host_use_path, 2 *
			                        numPaths * sizeof(unsigned int),
			                        cudaMemcpyHostToDevice));
		}

		if (num_emitters > 0) {
			HANDLE_ERROR(cudaMalloc((void**)(&x_emitters), numMPs *
			                        num_emitters * sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void**)(&y_emitters), numMPs *
			                        num_emitters * sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void**)(&z_emitters), numMPs *
			                        num_emitters * sizeof(int)));

			HANDLE_ERROR(cudaMemcpy((void*)x_emitters,
			                        geom_host.x_emitters[0],
			                        numMPs * num_emitters * sizeof(int),
			                        cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy((void*)y_emitters,
			                        geom_host.y_emitters[0],
			                        numMPs * num_emitters * sizeof(int),
			                        cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy((void*)z_emitters,
			                        geom_host.z_emitters[0],
			                        numMPs * num_emitters * sizeof(int),
			                        cudaMemcpyHostToDevice));
		}

		if (num_receivers > 0) {
			HANDLE_ERROR(cudaMalloc((void**)(&x_receivers), numMPs *
			                        num_receivers * sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void**)(&y_receivers), numMPs *
			                        num_receivers * sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void**)(&z_receivers), numMPs *
			                        num_receivers * sizeof(int)));

			HANDLE_ERROR(cudaMemcpy((void*)x_receivers,
			                        geom_host.x_receivers[0],
			                        numMPs * num_receivers * sizeof(int),
			                        cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy((void*)y_receivers,
			                        geom_host.y_receivers[0],
			                        numMPs * num_receivers * sizeof(int),
			                        cudaMemcpyHostToDevice));
			HANDLE_ERROR(cudaMemcpy((void*)z_receivers,
			                        geom_host.z_receivers[0],
			                        numMPs * num_receivers * sizeof(int),
			                        cudaMemcpyHostToDevice));
		}
	}
}

geometry_device::geometry_device(const geometry_device &geom_dev) :
	num_emitters(geom_dev.num_emitters),
	num_receivers(geom_dev.num_receivers), 
	rv_x(geom_dev.rv_x), rv_y( geom_dev.rv_y), rv_z(geom_dev.rv_z),
	scale_factor(geom_dev.scale_factor), 
	x_emitters(NULL), y_emitters(NULL), z_emitters(NULL), 
	x_receivers(NULL), y_receivers(NULL), z_receivers(NULL), 
	use_path(NULL), ld(geom_dev.ld), numPaths(geom_dev.numPaths),
	numMPs(geom_dev.numMPs), numSignals(geom_dev.numSignals) {

	if ( numMPs > 0) {

		if ((num_emitters > 0) & (num_receivers > 0) ) {

			HANDLE_ERROR(cudaMalloc((void**)&use_path, 2 *
			                        numPaths * sizeof(int)));
			HANDLE_ERROR(cudaMemcpy(use_path, geom_dev.use_path, 2 *
			                        numPaths * sizeof(int),
			                        cudaMemcpyDeviceToDevice));
		}

		if (num_emitters > 0) {

			HANDLE_ERROR(cudaMalloc((void**)(&x_emitters), numMPs *
			                        num_emitters * sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void**)(&y_emitters), numMPs *
			                        num_emitters * sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void**)(&z_emitters), numMPs *
			                        num_emitters * sizeof(int)));

			HANDLE_ERROR(cudaMemcpy((void*)x_emitters,
			                        geom_dev.x_emitters, numMPs *
			                        num_emitters * sizeof(int),
			                        cudaMemcpyDeviceToDevice));
			HANDLE_ERROR(cudaMemcpy((void*)y_emitters,
			                        geom_dev.y_emitters, numMPs *
			                        num_emitters * sizeof(int),
			                        cudaMemcpyDeviceToDevice));
			HANDLE_ERROR(cudaMemcpy((void*)z_emitters,
			                        geom_dev.z_emitters, numMPs *
			                        num_emitters * sizeof(int),
			                        cudaMemcpyDeviceToDevice));
		}

		if (num_receivers > 0) {
			HANDLE_ERROR(cudaMalloc((void**)(&x_receivers), numMPs *
			                        num_receivers * sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void**)(&y_receivers), numMPs *
			                        num_receivers * sizeof(int)));
			HANDLE_ERROR(cudaMalloc((void**)(&z_receivers), numMPs *
			                        num_receivers * sizeof(int)));

			HANDLE_ERROR(cudaMemcpy((void*)x_receivers,
			                        geom_dev.x_receivers, numMPs *
			                        num_receivers * sizeof(int),
			                        cudaMemcpyDeviceToDevice));
			HANDLE_ERROR(cudaMemcpy((void*)y_receivers,
			                        geom_dev.y_receivers, numMPs *
			                        num_receivers * sizeof(int),
			                        cudaMemcpyDeviceToDevice));
			HANDLE_ERROR(cudaMemcpy((void*)z_receivers,
			                        geom_dev.z_receivers, numMPs *
			                        num_receivers * sizeof(int),
			                        cudaMemcpyDeviceToDevice));
		}
	}
}

geometry_device::~geometry_device() {

	if (use_path != NULL) cudaFree(use_path); use_path = NULL;

	if (x_emitters != NULL) cudaFree(x_emitters); x_emitters = NULL;

	if (y_emitters != NULL) cudaFree(y_emitters); y_emitters = NULL;

	if (z_emitters != NULL) cudaFree(z_emitters); z_emitters = NULL;

	if (x_receivers != NULL) cudaFree(x_receivers); x_receivers = NULL;

	if (y_receivers != NULL) cudaFree(y_receivers); y_receivers = NULL;

	if (z_receivers != NULL) cudaFree(z_receivers); z_receivers = NULL;

}

#endif

#endif /* CONTAINER_DEVICE_H_ */
