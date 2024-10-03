#pragma once

#include "math.hpp"
#include "autograd.hpp"

namespace ml::regression
{
	using namespace ml::autograd;

	class linear : public differentiable
	{
	public:

		linear()
			: _y(),
			_X(),
			_b()
		{
		}

		linear(const matrix_t& y, const matrix_t& X)
			: _y(y),
			_X(X),
			_b(nd::array<>::random({ X.shape()[1], 1 }))
		{
		}

		parameter operator()(const std::vector<parameter>& params) const
		{
			auto& X = params[0];
			return X * _b;
		}


	private:
		parameter _y;
		parameter _X;
		parameter _b;
	};



	class logistic : public differentiable
	{
	public:

		logistic()
			: _y(),
			_X(),
			_w()
		{
		}

		logistic(const matrix_t& y, const matrix_t& X)
			: _y(y),
			_X(X),
			_w(nd::array<>::random({ X.shape()[1], 1 }))
		{
		}

		parameter operator()(const std::vector<parameter>& params) const
		{
			auto& X = params[0];
			return softmax(X * _w);
		}

	private:
		parameter _y;
		parameter _X;
		parameter _w;

		std::vector<size_t> _trainable_param_ids() const
		{
			return { _w.id() };
		}

		void _update_parameter(size_t id, const matrix_t& delta)
		{
			_w.set_value(_w.value() - delta);
		}
	};
}