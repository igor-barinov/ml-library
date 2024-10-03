#pragma once

#include "ndimensions/array.hpp"

namespace ml
{
	using matrix_t = nd::array<double>;

	inline matrix_t identity(size_t n) { return nd::array<double>::identity(n); }
	inline matrix_t ones(const nd::shape_t& shape) { return nd::array<double>::ones(shape); }
	inline matrix_t random(const nd::shape_t& shape) { return nd::array<double>::random(shape); }

	inline matrix_t pow(const matrix_t& X, double p)
	{
		auto fn = [p](double x)
			{
				return std::pow(x, p);
			};

		return X.map(fn);
	}

	inline matrix_t sqrt(const matrix_t& X) { return X.map(matrix_t::unary_fn{ &sqrtl }); }

	inline matrix_t d_sqrt(const matrix_t& X)
	{
		auto dx = [](double x)
			{
				return 1.0 / (2 * std::sqrt(x));
			};

		return X.map(dx);
	}

	inline matrix_t log(const matrix_t& X) { return X.map(logl);}

	inline matrix_t d_log(const matrix_t& X)
	{
		auto dx = [](double x)
			{
				return 1.0 / x;
			};

		return X.map(dx);
	}

	inline matrix_t exp(const matrix_t& X) { return X.map(expl); }

	inline matrix_t sigmoid(const matrix_t& X)
	{
		auto sig = [](double x)
			{
				return 1.0 / (1.0 + std::exp(-x));
			};

		return X.map(sig);
	}

	inline matrix_t d_sigmoid(const matrix_t& X)
	{
		auto d_sig = [](double x)
			{
				float sx = 1.0 / (1.0 + std::exp(-x));
				return sx * (1.0 - sx);
			};

		return X.map(d_sig);
	}

	inline matrix_t softmax(const matrix_t& X)
	{
		matrix_t stableX = X - X.max();
		matrix_t expVals = exp(stableX);
		double sum = expVals.sum();

		return expVals / sum;
	}

	inline matrix_t d_softmax(const matrix_t& X)
	{
		size_t N = X.shape()[0];
		matrix_t S = softmax(X);
		matrix_t S_vec = S({ nd::range(N), nd::range(1) });
		matrix_t S_mat = S_vec;
		for (size_t i = 1; i < S.shape()[0]; ++i)
		{
			S_mat = S_mat.concat(S_vec, 1);
		}

		matrix_t S_diag = matrix_t::from_diag(S_vec.squeeze());

		return S_diag - (S_mat * S_mat.T());
	}

	inline matrix_t relu(const matrix_t& X)
	{
		auto fn = [](double x)
			{
				return (x > 0.0) ? x : 0.0;
			};

		return X.map(fn);
	}

	inline matrix_t d_relu(const matrix_t& X)
	{
		auto fn = [](double x)
			{
				return (x > 0.0) ? 1.0 : 0.0;
			};

		return X.map(fn);
	}

	inline matrix_t sin(const matrix_t& X) { return X.map(sinl); }

	inline matrix_t cos(const matrix_t& X) { return X.map(cosl); }

	inline matrix_t tan(const matrix_t& X) { return X.map(tanl); }

	inline matrix_t sec(const matrix_t& X)
	{
		auto fn = [](double x)
			{
				return 1.0 / std::cos(x);
			};

		return X.map(fn);
	}

	inline matrix_t csc(const matrix_t& X)
	{
		auto fn = [](double x)
			{
				return 1.0 / std::sin(x);
			};

		return X.map(fn);
	}

	inline matrix_t d_sin(const matrix_t& X) { return cos(X); }

	inline matrix_t d_cos(const matrix_t& X) { return -1 * sin(X); }

	inline matrix_t d_tan(const matrix_t& X)
	{
		auto fn = [](double x)
			{
				double c = std::cos(x);
				return 1.0 / (c * c);
			};

		return X.map(fn);
	}
}