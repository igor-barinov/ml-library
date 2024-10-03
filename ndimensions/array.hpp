#pragma once

#include "utils.hpp"
#include "array_iter.hpp"

#include "mklutils.hpp"
#include <mkl/mkl_cblas.h>
#include <mkl/mkl_lapacke.h>

#include <vector>
#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <memory>
#include <functional>

namespace nd
{
	template <typename Ty = double>
	class array
	{
	public:

		/*
		* TYPDEFS
		*/

		using ndarray_t = array<Ty>;
		using iterator = array_iterator<Ty>;
		using mkl_props_t = mkl_props<Ty>;
		typedef std::function<Ty(Ty)> unary_fn;



		/*
		* CTORS
		*/

		void swap(ndarray_t& other)
		{
			using std::swap;
			swap(_values, other._values);
			swap(_nItems, other._nItems);
			swap(_shape, other._shape);
			swap(_shapeHash, other._shapeHash);
			swap(_strides, other._strides);
		}

		array()
			: _values(nullptr),
			_nItems(0),
			_shape(),
			_shapeHash(0),
			_strides()
		{
		}

		array(const shape_t& shape)
			: _values(nullptr),
			_nItems(0),
			_shape(shape),
			_shapeHash(std::hash<shape_t>()(shape)),
			_strides(shape.size())
		{
			_alloc();
		}

		array(const shape_t& shape, const Ty& fillValue)
			: array(shape)
		{
			std::fill(_values, _values + _nItems, fillValue);
		}

		array(Ty scalar)
			: array({1}, scalar)
		{
		}

		array(const ndarray_t& other)
			: ndarray_t(other._shape)
		{
			memcpy(_values, other._values, sizeof(Ty) * other._nItems);
		}

		array(ndarray_t&& other) noexcept
			: ndarray_t()
		{
			swap(other);
		}

		ndarray_t& operator=(ndarray_t other)
		{
			swap(other);
			return *this;
		}

		~array()
		{
			_free();
		}



		/*
		* ARRAY PROPERTIES
		*/

		inline const shape_t& shape() const { return _shape; }

		inline size_t dims() const { return _shape.size(); }

		inline size_t N() const { return _nItems; }

		inline bool empty() const { return _values == nullptr || _nItems == 0; }

		inline bool scalar() const { return _shape.size() == 1 && _shape[0] == 1; }

		inline bool vector() const { return _shape.size() == 1; }

		inline bool matrix() const { return _shape.size() == 2; }

		inline bool square() const { return matrix() && _shape[0] == _shape[1]; }

		inline index_t zero_index() const { return index_t(_shape.size(), 0); }



		/*
		* ARITHMETIC OPERATIONS
		*/

		ndarray_t operator+(const ndarray_t& other) const
		{
			if (!_same_shape_as(other)) { throw std::invalid_argument("Cannot add arrays with different shapes"); }

			ndarray_t result(_shape);
			for (size_t i = 0; i < _nItems; ++i)
			{
				result._values[i] = _values[i] + other._values[i];
			}
			return result;
		}

		ndarray_t& operator+=(const ndarray_t& other)
		{
			if (!_same_shape_as(other)) { throw std::invalid_argument("Cannot add arrays with different shapes"); }

			for (size_t i = 0; i < _nItems; ++i)
			{
				_values[i] += other._values[i];
			}
			return *this;
		}

		ndarray_t operator-(const ndarray_t& other) const
		{
			if (!_same_shape_as(other)) { throw std::invalid_argument("Cannot subtract arrays with different shapes"); }

			ndarray_t result(_shape);
			for (size_t i = 0; i < _nItems; ++i)
			{
				result._values[i] = _values[i] - other._values[i];
			}
			return result;
		}

		ndarray_t& operator-=(const ndarray_t& other)
		{
			if (!_same_shape_as(other)) { throw std::invalid_argument("Cannot subtract arrays with different shapes"); }

			for (size_t i = 0; i < _nItems; ++i)
			{
				_values[i] -= other._values[i];
			}
			return *this;
		}

		ndarray_t operator*(const ndarray_t& other) const
		{
			if (scalar() && other.scalar())
			{
				return ndarray_t(_values[0] * other._values[0]);
			}

			if (scalar())
			{
				return other * _values[0];
			}

			if (other.scalar())
			{
				return *this * other._values[0];
			}

			if (vector() && other.vector())
			{
				return dot(other);
			}

			if (!matrix() || !other.matrix()) { throw std::invalid_argument("Cannot multiply arrays with more than 2 dimensions"); }
			if (_shape[1] != other._shape[0]) { throw std::invalid_argument("A * B requries the shape of A to be [a, b] and the shape of B to be [b, c]"); }

			ndarray_t result({ _shape[0], other._shape[1] });
			double alpha = 1.0;
			double beta = 0.0;

			mkl_props_t propsA(*this);
			mkl_props_t propsB(other);
			mkl_props_t propsC(result);

			Ty* p = result._values;

			cblas_dgemm(
				CblasColMajor,
				CblasNoTrans,
				CblasNoTrans,
				propsA.m,
				propsC.n,
				propsA.n,
				alpha,
				propsA.data,
				propsA.ld,
				propsB.data,
				propsB.ld,
				beta,
				propsC.data,
				propsC.ld
			);

			return result;
		}

		Ty dot(const ndarray_t& other) const
		{
			if (_nItems != other._nItems) { throw std::invalid_argument("Cannot take dot product of arrays of different length"); }

			Ty sum{};
			for (size_t i = 0; i < _nItems; ++i)
			{
				sum += _values[i] * other._values[i];
			}
			return sum;
		}

		ndarray_t hadamard(const ndarray_t& other) const
		{
			if (!_same_shape_as(other)) { throw std::invalid_argument("Cannot multiply arrays with different shapes"); }

			ndarray_t result(_shape);
			for (size_t i = 0; i < _nItems; ++i)
			{
				result._values[i] = _values[i] * other._values[i];
			}
			return result;
		}

		ndarray_t T() const
		{
			if (!matrix()) { throw std::invalid_argument("Array is not a matrix"); }

			ndarray_t transpose({ _shape[1], _shape[0] });

			for (size_t i = 0; i < _shape[0]; ++i)
			{
				for (size_t j = 0; j < _shape[1]; ++j)
				{
					size_t srcOffset = offset_of({ i, j }, _strides);
					size_t destOffset = offset_of({ j, i }, transpose._strides);
					transpose._values[destOffset] = _values[srcOffset];
				}
			}

			return transpose;
		}

		ndarray_t inv()
		{
			if (!square()) { throw std::invalid_argument("Cannot inverse a non-square matrix"); }

			ndarray_t inverse(*this);

			mkl_props_t props(inverse);
			std::unique_ptr<int> ipiv(new int[props.m]);
			LAPACKE_dgetrf(LAPACK_COL_MAJOR, props.m, props.m, props.data, props.ld, ipiv.get());
			LAPACKE_dgetri(LAPACK_COL_MAJOR, props.m, props.data, props.ld, ipiv.get());
			return inverse;
		}

		ndarray_t map(unary_fn transform) const
		{
			ndarray_t result(_shape);
			for (size_t i = 0; i < _nItems; ++i)
			{
				result._values[i] = transform(_values[i]);
			}
			return result;
		}

		ndarray_t operator+(Ty scalar) const
		{
			ndarray_t result(*this);
			for (size_t i = 0; i < _nItems; ++i)
			{
				result._values[i] += scalar;
			}
			return result;
		}

		ndarray_t& operator+=(Ty scalar)
		{
			for (size_t i = 0; i < _nItems; ++i)
			{
				_values[i] += scalar;
			}
			return *this;
		}

		inline friend ndarray_t operator+(Ty scalar, const ndarray_t& X) { return X + scalar; }

		ndarray_t operator-(Ty scalar) const
		{
			ndarray_t result(*this);
			for (size_t i = 0; i < _nItems; ++i)
			{
				result._values[i] -= scalar;
			}
			return result;
		}

		ndarray_t& operator-=(Ty scalar)
		{
			for (size_t i = 0; i < _nItems; ++i)
			{
				_values[i] -= scalar;
			}
			return *this;
		}

		friend ndarray_t operator-(Ty scalar, const ndarray_t& X)
		{
			ndarray_t result(X);
			for (size_t i = 0; i < result._nItems; ++i)
			{
				result._values[i] = scalar - X._values[i];
			}
			return result;
		}

		ndarray_t operator*(Ty scalar) const
		{
			ndarray_t result(*this);
			for (size_t i = 0; i < _nItems; ++i)
			{
				result._values[i] *= scalar;
			}
			return result;
		}

		ndarray_t& operator*=(Ty scalar)
		{
			for (size_t i = 0; i < _nItems; ++i)
			{
				_values[i] *= scalar;
			}
			return *this;
		}

		inline friend ndarray_t operator*(Ty scalar, const ndarray_t& X) { return X * scalar; }

		ndarray_t operator/(Ty scalar) const
		{
			ndarray_t result(*this);
			for (size_t i = 0; i < _nItems; ++i)
			{
				result._values[i] /= scalar;
			}
			return result;
		}

		ndarray_t& operator/=(Ty scalar)
		{
			for (size_t i = 0; i < _nItems; ++i)
			{
				_values[i] /= scalar;
			}
			return *this;
		}

		friend ndarray_t operator/(Ty scalar, const ndarray_t& X)
		{
			ndarray_t result(X);
			for (size_t i = 0; i < result._nItems; ++i)
			{
				result._values[i] = scalar / X._values[i];
			}
			return result;
		}



		/*
		* STATISTICS
		*/

		Ty sum() const
		{
			Ty result{};
			for (size_t i = 0; i < _nItems; ++i)
			{
				result += _values[i];
			}
			return result;
		}

		ndarray_t sum(size_t dimension) const
		{
			if (dimension >= _shape.size()) { throw std::invalid_argument("Cannot take sum along dimension"); }

			shape_t resultShape(_shape);
			resultShape[dimension] = 1;
			ndarray_t result(resultShape);

			std::vector<range> ndRange = bounds_of(_shape);

			for (size_t n = 0; n < _shape[dimension]; ++n)
			{
				ndRange[dimension] = range(n, n + 1);
				result += (*this)(ndRange);
			}

			return result;
		}

		Ty max() const
		{
			Ty result = _values[0];
			for (size_t i = 1; i < _nItems; ++i)
			{
				result = (_values[i] > result) ? _values[i] : result;
			}

			return result;
		}

		Ty mean() const
		{
			return sum() / static_cast<Ty>(_nItems);
		}

		ndarray_t mean(size_t dimension) const
		{
			if (dimension >= _shape.size()) { throw std::invalid_argument("Cannot take mean along dimension"); }

			return sum(dimension) / static_cast<Ty>(_shape[dimension]);
		}

		Ty variance() const
		{
			Ty u = mean();
			Ty sum{};
			for (size_t i = 0; i < _nItems; ++i)
			{
				Ty diff = _values[i] - u;
				sum += diff * diff;
			}
			return sum / static_cast<Ty>(_nItems);
		}

		ndarray_t variance(size_t dimension) const
		{
			if (dimension >= _shape.size()) { throw std::invalid_argument("Cannot take variance along dimension"); }

			ndarray_t u = mean(dimension);
			ndarray_t sum(u.shape);
			std::vector<range> ndRange = bounds_of(_shape);
			for (size_t i = 0; i < _shape[dimension]; ++i)
			{
				ndRange[dimension] = i;

				ndarray_t slice = (*this)(ndRange);
				ndarray_t diff = slice - u;
				sum += diff.hadamard(diff);
			}

			return sum / static_cast<Ty>(_shape[dimension]);
		}

		inline Ty stddev() const { return std::sqrt(variance()); }

		inline Ty stddev(size_t dimension) const { return variance(dimension).map(std::sqrt); }



		/*
		* GENERATED ARRAYS
		*/

		inline static ndarray_t zeros(const shape_t& shape) { return ndarray_t(shape); }

		inline static ndarray_t ones(const shape_t& shape) { return ndarray_t(shape, static_cast<Ty>(1)); }

		static ndarray_t identity(size_t n)
		{
			ndarray_t I({ n, n });
			for (size_t i = 0; i < n; ++i)
			{
				I({ i,i }) = static_cast<Ty>(1.0f);
			}

			return I;
		}

		static ndarray_t random(const shape_t& shape)
		{
			ndarray_t mat(shape);
			for (size_t i = 0; i < mat._nItems; ++i)
			{
				mat._values[i] = random_uniform();
			}

			return mat;
		}
		
		static ndarray_t from_diag(const ndarray_t& diagonal)
		{
			if (diagonal.dims() != 1) { throw std::invalid_argument("Supplied diagonal must be a vector"); }
			
			size_t N = diagonal.N();
			ndarray_t mat({ N, N });
			for (size_t i = 0; i < N; ++i)
			{
				mat({ i, i }) = diagonal._values[i];
			}

			return mat;
		}



		/*
		* CONST ELEMENT ACCESS
		*/

		inline Ty operator()(const index_t& ndIndex) const { return _values[offset_of(ndIndex, _strides)]; }

		Ty at(const index_t& ndIndex) const
		{
			_throw_if_invalid(ndIndex);
			return _values[offset_of(ndIndex, _strides)];
		}

		ndarray_t operator()(const std::vector<range>& ndRange) const
		{
			shape_t sliceShape(ndRange.size());
			index_t i = index_t(ndRange.size());

			for (size_t n = 0; n < ndRange.size(); ++n)
			{
				sliceShape[n] = ndRange[n].size();
				i[n] = ndRange[n].start;
			}

			ndarray_t slice(sliceShape);
			size_t srcOffset = offset_of(i, _strides);
			size_t destOffset = 0;
			while (destOffset < slice._nItems)
			{
				slice._values[destOffset] = _values[srcOffset];
				increment_index(i, ndRange);
				srcOffset = offset_of(i, _strides);
				destOffset++;
			}

			return slice;
		}

		ndarray_t slice(const std::vector<range>& ndRange) const
		{
			if (ndRange.size() != _shape.size()) { throw std::invalid_argument("Range has incorrect number of dimensions"); }
			for (size_t n = 0; n < ndRange.size(); ++n)
			{
				if (ndRange[n].size() == 0) { throw std::invalid_argument("Range must have non-zero size"); }
				if (ndRange[n].start >= _shape[n] || ndRange[n].end >= _shape[n]) { throw std::invalid_argument("Range is out of bounds at dimension " + n); }
			}

			return (*this)(ndRange);
		}



		/*
		* ELEMENT ACCESS
		*/

		Ty& operator()(const index_t& ndIndex) { return _values[offset_of(ndIndex, _strides)]; }

		Ty& at(const index_t& ndIndex)
		{
			_throw_if_invalid(ndIndex);
			return _values[offset_of(ndIndex, _strides)];
		}



		/*
		* TRANSFORMATIONS
		*/

		ndarray_t& reshape(const shape_t& newShape)
		{
			size_t newSize = std::reduce(newShape.begin(), newShape.end(), (size_t)1, std::multiplies<size_t>{});
			if (_nItems != newSize) { throw std::invalid_argument("New shape may not alter the number of items in array"); }

			_shape = newShape;
			_strides = calculate_strides(newShape);
			return *this;
		}

		ndarray_t concat(const ndarray_t& other, size_t dimension = 0) const 
		{
			if (dims() != other.dims()) { throw std::invalid_argument("Array being added must have same number dimensions"); }

			shape_t newShape(_shape);
			newShape[dimension] += other._shape[dimension];

			ndarray_t result(newShape);
			index_t destIndex = result.zero_index();
			size_t srcOffsetA = 0;
			size_t srcOffsetB = 0;
			size_t nValsCopied = 0;

			while (nValsCopied < result.N())
			{
				if (destIndex[dimension] < _shape[dimension])
				{
					result(destIndex) = _values[srcOffsetA];
					srcOffsetA++;
				}
				else
				{
					result(destIndex) = other._values[srcOffsetB];
					srcOffsetB++;
				}

				increment_index(destIndex, newShape);
				nValsCopied++;
			}

			return result;
		}

		ndarray_t& squeeze()
		{
			shape_t newShape;
			for (auto dimSize : _shape)
			{
				if (dimSize > 1)
				{
					newShape.push_back(dimSize);
				}
			}

			_strides = calculate_strides(newShape);
			_shape = newShape;
			return *this;
		}

		ndarray_t diag() const
		{
			if (!square()) { throw std::invalid_argument("Cannot take diagonal of non-square array"); }

			size_t N = _shape[0];
			ndarray_t diagonal({ N });
			for (size_t i = 0; i < N; ++i)
			{
				diagonal._values[i] = _values[offset_of({ i, i }, _strides)];
			}

			return diagonal;
		}



		/*
		* COMPARISON
		*/

		bool operator==(const ndarray_t& other) const
		{
			if (!_same_shape_as(other)) { return false; }

			for (size_t i = 0; i < _nItems; ++i)
			{
				if (_values[i] != other._values[i]) { return false; }
			}

			return true;
		}

		bool approx_equal(const ndarray_t& other, double eps = 0.0001) const
		{
			if (!_same_shape_as(other)) { return false; }

			for (size_t i = 0; i < _nItems; ++i)
			{
				if (std::abs(_values[i] - other._values[i]) > eps) { return false; }
			}

			return true;
		}



		/*
		* ITERATORS
		*/

		inline iterator begin() { return iterator(&_values[0]); }
		inline iterator end() { return iterator(&_values[_nItems]); }

	private:

		/*
		* FRIENDS
		*/

		friend struct mkl_props<Ty>;



		/*
		* PRIVATE MEMBERS
		*/

		Ty* _values;
		size_t _nItems;
		shape_t _shape;
		size_t _shapeHash;
		stride_t _strides;



		/*
		* RESOURCE METHODS
		*/

		void _alloc()
		{
			_nItems = std::reduce(_shape.begin(), _shape.end(), (size_t)1, std::multiplies<size_t>{});

			if (_nItems == 0)
			{
				_values = nullptr;
				return;
			}

			_values = new Ty[_nItems];
			memset(_values, 0, sizeof(Ty) * _nItems);
			_strides = calculate_strides(_shape);
		}

		void _free()
		{
			if (_values != nullptr)
			{
				delete[] _values;
				_values = nullptr;
				_nItems = 0;
			}
		}

		inline bool _same_shape_as(const ndarray_t& other) const { return _shapeHash == other._shapeHash; }

		void _throw_if_invalid(const index_t& ndIndex) const
		{
			if (ndIndex.size() != _shape.size()) { throw std::invalid_argument("Index does not have the correct number of dimensions"); }
			for (size_t n = 0; n < ndIndex.size(); ++n)
			{
				if (ndIndex[n] >= _shape[n]) { throw std::invalid_argument("Index exceeds bounds at dimension " + n); }
			}
		}
	};
}

namespace std
{
	template <typename T>
	void swap(nd::array<T>& arrA, nd::array<T>& arrB)
	{
		arrA.swap(arrB);
	}
}