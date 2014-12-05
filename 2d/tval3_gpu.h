#ifndef TVAL3_GPU_H_
#define TVAL3_GPU_H_

#include "container_host.h"
#include <complex>

#include "tval3_types.h"

extern const tval3_info tval3_gpu(mat_host &U, const sparse_mat_host &A,
                                  const mat_host &b,
                                  const tval3_options &opts,
                                  const mat_host &Ut = 0, bool pagelocked =
                                          true);

extern const tval3_info tval3_gpu(mat_host &U, const mat_host &A,
                                  const mat_host &b,
                                  const tval3_options &opts,
                                  const mat_host &Ut = 0, bool pagelocked =
                                          true);

#endif /* TVAL3_GPU_H_ */
