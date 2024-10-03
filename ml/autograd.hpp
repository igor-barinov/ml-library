#pragma once

#include "math.hpp"

/*
* https://github.com/mattjj/autodidact
*/

namespace ml::optimizers
{
	class SGD;
};

namespace ml::autograd
{
	class parameter
	{
	public:

		void swap(parameter& other)
		{
			using std::swap;

			swap(_value, other._value);
			swap(_id, other._id);
			swap(_parents, other._parents);
			swap(_partials, other._partials);
			swap(_gradFns, other._gradFns);
			swap(fnName, other.fnName);
		}

		parameter()
			: _value(),
			_id(_increment_id()),
			_parents(),
			_partials(),
			_gradFns()
		{
		}

		parameter(const matrix_t& value)
			: _value(value),
			_id(_increment_id()),
			_parents(),
			_partials(),
			_gradFns()
		{
		}

		parameter(const parameter& other)
			: _value(other._value),
			fnName(other.fnName),
			_id(other._id),
			_parents(other._parents),
			_partials(other._partials),
			_gradFns(other._gradFns)
		{
		}

		parameter(parameter&& other) noexcept
			: parameter()
		{
			swap(other);
		}

		parameter& operator=(parameter other)
		{
			swap(other);
			return *this;
		}

		const size_t id() const { return _id; }

		const matrix_t& value() const { return _value; }

		void set_value(const matrix_t& newVal)
		{
			_value = newVal;
		}

		matrix_t partial_wrt(size_t paramID) const
		{
			std::vector<std::pair<parameter, matrix_t>> stack = { {*this, ones(_value.shape())} };

			while (!stack.empty())
			{
				auto node = stack.back();
				stack.pop_back();

				if (node.first._id == paramID) { return node.second; }

				auto& parents = node.first._parents;
				for (size_t i = 0; i < parents.size(); ++i)
				{
					auto gradFn = node.first._gradFns[i];
					stack.push_back({ parents[i], gradFn(node.second, node.first._partials[i]) });
				}
			}
		}

		const std::vector<parameter>& parent_params() const { return _parents; }

		/*
		* ARITHMETIC DERIVATIVES
		*/

		parameter operator+(const parameter& other) const
		{
			parameter result;
			result.fnName = "mat + mat";
			result._value = _value + other._value;
			result._parents = { *this, other };
			result._partials = { ones(_value.shape()), ones(other._value.shape()) };
			result._gradFns = { _default_grad_fn, _default_grad_fn };
			return result;
		}

		parameter operator-(const parameter& other) const
		{
			parameter result;
			result.fnName = "mat - mat";
			result._value = _value - other._value;
			result._parents = { *this, other };
			result._partials = { ones(_value.shape()), ones(other._value.shape()) * -1.0 };
			result._gradFns = { _default_grad_fn, _default_grad_fn };
			return result;
		}

		parameter operator*(const parameter& other) const
		{
			parameter result;
			result.fnName = "mat * mat";
			result._value = _value * other._value;
			result._parents = { *this, other };

			if (_value.matrix() && other._value.matrix())
			{
				result._partials = { other._value.T(), _value.T()};
				auto gradFn1 = [](const matrix_t& dzdy, const matrix_t& dydx)
					{
						return dzdy * dydx;
					};

				auto gradFn2 = [](const matrix_t& dzdy, const matrix_t& dydx)
					{
						return dydx * dzdy;
					};

				result._gradFns = { gradFn1, gradFn2 };
			}
			else
			{
				result._partials = { other._value, _value };
				auto gradFn = [](const matrix_t& dzdy, const matrix_t& dydx)
					{
						return dzdy * dydx;
					};

				result._gradFns = { gradFn, gradFn };
			}

			return result;
		}

		parameter dot(const parameter& other) const
		{

			// y = this * other
			// dy/dthis = dthis/dx * other

			parameter result;
			result.fnName = "dot";
			result._value = _value.dot(other._value);
			result._parents = { *this, other };
			result._partials = { other._value, _value };

			auto gradFn = [](const matrix_t& dzdy, const matrix_t& dydx)
				{
					return dzdy.dot(dydx);
				};

			result._gradFns = { _default_grad_fn, _default_grad_fn };
			return result;
		}

		parameter hadamard(const parameter& other) const
		{
			parameter result;
			result.fnName = "hadamard";
			result._value = _value.hadamard(other._value);
			result._parents = { *this, other };
			result._partials = { other._value, _value };
			result._gradFns = { _default_grad_fn, _default_grad_fn };
			return result;
		}

		friend parameter operator-(double scalar, const parameter& X)
		{
			parameter result;
			result.fnName = "scalar - mat";
			result._value = scalar - X._value;
			result._parents = { X };
			result._partials = { -1.0 * ones(X._value.shape()) };
			result._gradFns = { _default_grad_fn };
			return result;
		}

		parameter operator*(double scalar) const
		{
			parameter result;
			result.fnName = "mat * scalar";
			result._value = _value * scalar;
			result._parents = { *this };
			result._partials = { scalar };

			auto gradFn = [](const matrix_t& dzdy, const matrix_t& dydx)
				{
					return dzdy * dydx.at({ 0 });
				};
			result._gradFns = { gradFn };
			return result;
		}

		friend parameter operator*(double scalar, const parameter& X) { return X * scalar; }

		parameter operator/(double scalar) const
		{
			parameter result;
			result.fnName = "mat / scalar";
			result._value = _value / scalar;
			result._parents = { *this };
			result._partials = { 1.0 / scalar };

			auto gradFn = [](const matrix_t& dzdy, const matrix_t& dydx)
				{
					return dzdy * dydx.at({ 0 });
				};
			result._gradFns = { _default_grad_fn };
			return result;
		}

		friend parameter operator/(double scalar, const parameter& X)
		{
			parameter result;
			result.fnName = "scalar / mat";
			result._value = scalar / X._value;
			result._parents = { X };
			auto xsq = X._value.hadamard(X._value);
			result._partials = { -1 / xsq };

			auto gradFn = [](const matrix_t& dzdy, const matrix_t& dydx)
				{
					return dzdy * dydx.at({ 0 });
				};
			result._gradFns = { _default_grad_fn };
			return result;
		}

		/*
		* MISC DERIVATIVES
		*/
		parameter T() const
		{
			parameter result;
			result.fnName = "T";
			result._value = _value.T();
			result._parents = { *this };
			result._partials = { _value };
			result._gradFns = { _default_grad_fn };
		}

	private:
		typedef matrix_t(*_grad_fn)(const matrix_t&, const matrix_t&);

		static matrix_t _default_grad_fn(const matrix_t& dzdy, const matrix_t& dydx)
		{
			return dzdy.hadamard(dydx);
		}
		
		size_t _id;
		matrix_t _value;
		const char* fnName;
		std::vector<parameter> _parents;
		std::vector<matrix_t> _partials;
		std::vector<_grad_fn> _gradFns;

		size_t _increment_id()
		{
			static size_t counter = 0;
			return counter++;
		}

		/*
		* MATH FUNCTION DERIVATIVES
		*/

		friend parameter sqrt(const parameter& X)
		{
			parameter result;
			result.fnName = "sqrt";
			result._value = sqrt(X._value);
			result._parents = { X };
			result._partials = { d_sqrt(X._value) };
			result._gradFns = { parameter::_default_grad_fn };
			return result;
		}

		friend parameter exp(const parameter& X)
		{
			parameter result;
			result.fnName = "exp";
			result._value = exp(X._value);
			result._parents = { X };
			result._partials = { exp(X._value) };
			result._gradFns = { parameter::_default_grad_fn };
			return result;
		}

		friend parameter log(const parameter& X)
		{
			parameter result;
			result.fnName = "log";
			result._value = log(X._value);
			result._parents = { X };
			result._partials = { d_log(X._value) };
			result._gradFns = { parameter::_default_grad_fn };
			return result;
		}

		friend parameter sin(const parameter& X)
		{
			parameter result;
			result.fnName = "sin";
			result._value = sin(X._value);
			result._parents = { X };
			result._partials = { d_sin(X._value) };
			result._gradFns = { parameter::_default_grad_fn };
			return result;
		}

		friend parameter cos(const parameter& X)
		{
			parameter result;
			result.fnName = "cos";
			result._value = cos(X._value);
			result._parents = { X };
			result._partials = { d_cos(X._value) };
			result._gradFns = { parameter::_default_grad_fn };
			return result;
		}

		friend parameter tan(const parameter& X)
		{
			parameter result;
			result.fnName = "tan";
			result._value = tan(X._value);
			result._parents = { X };
			result._partials = { d_tan(X._value) };
			result._gradFns = { parameter::_default_grad_fn };
			return result;
		}

		friend parameter sigmoid(const parameter& X)
		{
			parameter result;
			result.fnName = "sigmoid";
			result._value = sigmoid(X._value);
			result._parents = { X };
			result._partials = { d_sigmoid(X._value) };
			result._gradFns = { parameter::_default_grad_fn };
			return result;
		}
		
		friend parameter softmax(const parameter& X)
		{
			parameter result;
			result.fnName = "softmax";
			result._value = softmax(X._value);
			result._parents = { X };
			result._partials = { d_softmax(X._value) };
			result._gradFns = { parameter::_default_grad_fn };
			return result;
		}
	};


	class differentiable
	{
	public:
		virtual parameter operator()(const std::vector<parameter>& params) const = 0;

	protected:
		virtual std::vector<size_t> _trainable_param_ids() const { return {}; }
		virtual void _update_parameter(size_t id, const matrix_t& delta) {}

		friend class ml::optimizers::SGD;
	};
}