#ifndef _TRACK_ELLIPSE_KERNEL_H_
#define _TRACK_ELLIPSE_KERNEL_H_

#include <meschach/matrix.h>

// Compatibility macros for old meschach API
#ifndef m_get_val
#define m_get_val(A,i,j)  m_entry((A),(i),(j))
#endif
#ifndef v_get_val
#define v_get_val(x,i)    v_entry((x),(i))
#endif

#ifdef __cplusplus
extern "C" {
#endif
extern void IMGVF_cuda_init(MAT **I, int Nc);
extern void IMGVF_cuda_cleanup(MAT **IMGVF_out, int Nc);
extern void IMGVF_cuda(MAT **I, MAT **IMGVF, double vx, double vy, double e, int max_iterations, double cutoff, int Nc);
#ifdef __cplusplus
}
#endif


#endif
