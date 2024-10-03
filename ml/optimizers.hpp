#pragma once

#include "metrics.hpp"

namespace ml::optimizers
{
	using namespace ml::autograd;
	using namespace ml::metrics;

	

	class SGD
	{
	public:

		SGD(cost_function costFn, double learningRate = 0.05, size_t maxIterations = 100)
			: _lr(learningRate),
			_maxIter(maxIterations),
			_costFn(costFn)
		{
		}

		void optimize(differentiable& model, const std::vector<parameter>& inputs, const matrix_t& y)
		{
			for (size_t i = 0; i < _maxIter; ++i)
			{
				parameter yhat = model(inputs);
				parameter cost = _costFn(y, yhat);

				for (auto& id : model._trainable_param_ids())
				{
					matrix_t grad = cost.partial_wrt(id);
					model._update_parameter(id, grad * _lr);
				}
			}
			
			
		}

	private:
		double _lr;
		size_t _maxIter;
		cost_function _costFn;
	};
}