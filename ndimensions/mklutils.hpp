#pragma once

#include "definitions.hpp"

template <typename T>
struct mkl_props
{
	int m;
	int n;
	int ld;
	double* data;
	int info;

	mkl_props(const nd::array<T>& ndarray)
		: m(static_cast<int>(ndarray._shape[0])),
		n(static_cast<int>(ndarray._shape[1])),
		ld(static_cast<int>(ndarray._shape[0])),
		data(ndarray._values),
		info(0)
	{
	}
};