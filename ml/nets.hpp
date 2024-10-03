#pragma once

#include "layers.hpp"

namespace ml::nets
{
	class mlp
	{
	public:

		mlp(const std::vector<size_t>& layerSizes, layers::activation_fn activation = sigmoid)
			: _layers(layerSizes.size())
		{
			_layers[0] = layers::dense({ 1, layerSizes[0] });
			for (size_t i = 1; i < _layers.size(); ++i)
			{
				size_t nInputs = _layers[i - 1].size();
				_layers[i] = layers::dense({ layerSizes[i], nInputs }, activation);
			}
		}

	private:
		matrix_t _X;
		std::vector<ml::layers::dense> _layers;

		matrix_t _feed_forward(const matrix_t& X)
		{
			_layers[0].resize({ _layers[0].size(), X.shape()[0] });

			matrix_t prevLayer = X;
			for (auto& layer : _layers)
			{
				prevLayer = layer(prevLayer);
			}

			return prevLayer;
		}
	};
}