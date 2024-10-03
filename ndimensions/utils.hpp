#pragma once

#include <vector>
#include <stdexcept>
#include <random>

namespace nd
{
	typedef std::vector<size_t> shape_t;
	typedef std::vector<size_t> stride_t;
	typedef std::vector<size_t> index_t;

	struct range
	{
		size_t start;
		size_t end;
		size_t steps;

		range() : start(0), end(0), steps(1) {}
		range(size_t n) : start(0), end(n), steps(1) {}
		range(size_t start, size_t end, size_t steps = 1) : start(start), end(end), steps(steps)
		{
			if (start > end) { throw std::invalid_argument("Start of range cannot be greater than end"); }
		}

		inline size_t size() const { return (end - start + steps - 1) / steps; }
	};

	inline range operator ""_r(unsigned long long n) { return range(n); }

	inline stride_t calculate_strides(const shape_t& shape)
	{
		stride_t strides(shape.size());
		size_t strideProduct = 1;
		for (size_t i = 0; i < shape.size(); ++i)
		{
			strides[i] = strideProduct;
			strideProduct *= shape[i];
		}

		return strides;
	}

	inline size_t offset_of(const index_t& ndIndex, const stride_t& strides)
	{
		size_t offset = 0;
		for (size_t i = 0; i < ndIndex.size(); ++i)
		{
			offset += ndIndex[i] * strides[i];
		}

		return offset;
	}

	inline std::vector<range> bounds_of(const shape_t& shape)
	{
		std::vector<range> bounds(shape.size());
		for (size_t i = 0; i < bounds.size(); ++i)
		{
			bounds[i].end = shape[i];
		}

		return bounds;
	}

	inline index_t start_index(const std::vector<range>& ndRange)
	{
		index_t index(ndRange.size());
		for (size_t n = 0; n < ndRange.size(); ++n)
		{
			index[n] = ndRange[n].start;
		}
		return index;
	}

	inline void increment_index(index_t& ndIndex, const shape_t& shape)
	{
		ndIndex[0]++;
		for (size_t n = 1; n < shape.size(); ++n)
		{
			if (ndIndex[n - 1] >= shape[n - 1])
			{
				ndIndex[n - 1] = 0;
				ndIndex[n]++;
			}
		}
	}

	inline void increment_index(index_t& ndIndex, const std::vector<range>& ndRange)
	{
		ndIndex[0] += ndRange[0].steps;
		for (size_t n = 1; n < ndRange.size(); ++n)
		{
			if (ndIndex[n - 1] >= ndRange[n - 1].end)
			{
				ndIndex[n - 1] = ndRange[n - 1].start;
				ndIndex[n] += ndRange[n].steps;
			}
		}
	}

	inline double random_uniform()
	{
		static std::default_random_engine e;
		static std::uniform_real_distribution<> dist(0, 1);
		return dist(e);
	}

}

template <>
struct std::hash<nd::shape_t>
{
	size_t operator()(const nd::shape_t& s) const
	{
		using std::hash;

		size_t result = s.size();
		for (auto n : s)
		{
			result ^= hash<size_t>()(n) + 0x9e3779b9 + (result << 6) + (result >> 2);
		}

		return result;
	}
};