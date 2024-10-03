#pragma once

#include "autograd.hpp"

namespace ml::metrics
{
	using namespace ml::autograd;

	typedef parameter(*cost_function)(const parameter&, const parameter&);

	inline parameter cross_entropy(const parameter& y, const parameter& yhat)
	{
		return -1.0 * (y.hadamard(log(yhat)) + (1.0 - y).hadamard(log(1.0 - yhat)));
	}

	class metrics
	{
	};
}