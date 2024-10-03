#pragma once

#include "math.hpp"

namespace ml::layers
{
	typedef matrix_t(*activation_fn)(const matrix_t&);

	class dense
	{
	public:

		dense()
		{
		}

		dense(const nd::shape_t& shape, activation_fn activation = sigmoid)
			: _W(nd::array<>::random(shape)),
			_f(activation)
		{
		}

		inline size_t size() const { return _W.shape()[0]; }

		inline matrix_t operator()(const matrix_t& X) const { return _f(_add_biases(_W) * X); }

		void resize(const nd::shape_t& shape)
		{
			_W = nd::array<>::random(shape);
		}

	private:
		matrix_t _W;
		activation_fn _f;

		matrix_t _add_biases(const matrix_t& X) const
		{
			matrix_t ones({ 1, X.shape()[1] });
			return X.concat(ones);
		}
	};
}