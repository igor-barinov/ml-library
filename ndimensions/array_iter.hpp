#pragma once

#include "utils.hpp"

#include <iterator>
#include <cstddef>
#include <numeric>

namespace nd
{

	template <typename T>
	class array_iterator
	{
	public:

		using iterator = array_iterator<T>;
		using iterator_category = std::random_access_iterator_tag;
		using difference_type = std::ptrdiff_t;
		using value_type = T;
		using pointer = T*;
		using reference = T&;

		array_iterator(T* pValue)
			: _pValue(pValue)
		{
		}

		array_iterator(const iterator& other)
			: _pValue(other._pValue)
		{
		}

		inline reference operator*() { return *_pValue; }

		inline pointer operator->() { return _pValue; }

		inline reference operator[](difference_type diff) { return _pValue[diff]; }

		iterator& operator++()
		{
			_pValue++;
			return *this;
		}

		iterator operator++(int)
		{
			iterator tmp = *this;
			_pValue++;
			return tmp;
		}

		iterator& operator--()
		{
			_pValue--;
			return *this;
		}

		iterator operator--(int)
		{
			iterator tmp = *this;
			_pValue--;
			return tmp;
		}

		iterator& operator+=(difference_type diff)
		{
			_pValue += diff;
			return *this;
		}

		iterator& operator-=(difference_type diff)
		{
			_pValue -= diff;
			return *this;
		}

		iterator operator+(difference_type diff) const
		{
			iterator tmp(*this);
			tmp += diff;
			return tmp;
		}

		iterator operator-(difference_type diff) const
		{
			iterator tmp(*this);
			tmp -= diff;
			return tmp;
		}

		inline difference_type operator-(const iterator& other) const {	return _pValue - other._pValue;	}

		inline friend iterator operator+(difference_type diff, const iterator& iter) { return iter + diff; }

		inline friend iterator operator-(difference_type diff, const iterator& iter)
		{
			iterator tmp(iter);
			tmp._pValue = diff - tmp._pValue;
			return tmp;
		}

		inline friend bool operator== (const iterator& iterA, const iterator& iterB) { return iterA._pValue == iterB._pValue; }

		inline friend bool operator!= (const iterator& iterA, const iterator& iterB) { return !(iterA == iterB); }

		inline bool operator>(const iterator& other) const { return _pValue > other._pValue; }

		bool operator>=(const iterator& other) const { return _pValue >= other._pValue; }

		bool operator<(const iterator& other) const { return _pValue < other._pValue; }

		bool operator<=(const iterator& other) const { return _pValue <= other._pValue; }

	private:

		T* _pValue;
	};
}