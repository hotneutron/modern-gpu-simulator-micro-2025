#ifndef MISC_MATH_H
#define MISC_MATH_H

#include <meschach/matrix.h>

// Compatibility macros for old meschach API
#ifndef m_get_val
#define m_get_val(A,i,j)  m_entry((A),(i),(j))
#endif
#ifndef v_get_val
#define v_get_val(x,i)    v_entry((x),(i))
#endif

#define PI 3.14159


extern int double_eq(double f1, double f2);
extern MAT * gradient_x(MAT * input);
extern MAT * gradient_y(MAT * input);
extern double mean(VEC * in);
extern double std_dev(VEC * in);

#endif
