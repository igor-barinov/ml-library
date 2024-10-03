#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <unordered_set>

#include "ndimensions/array.hpp"

namespace data
{
	struct csv_props
	{
		bool ignoreHeader;
		std::vector<size_t> excludedCols;
	};

	struct default_column_parser
	{
		double operator()(const std::string& header, const std::string& value)
		{
			return std::stod(value);
		}
	};

	template <typename T, class ColParser = default_column_parser>
	nd::array<T> read_csv(const std::string& filepath, csv_props props, ColParser columnParser)
	{
		std::ifstream csvFile(filepath);
		std::string line;
		

		std::unordered_set<size_t> columnsToExclude(props.excludedCols.begin(), props.excludedCols.end());
		std::vector<std::vector<std::string>> cells;

		while (std::getline(csvFile, line))
		{
			std::stringstream linestream(line);
			linestream >> std::ws;

			cells.push_back(std::vector<std::string>{});
			bool quoteIsOpen = false;
			std::string cell;
			while (linestream.good())
			{
				char c;
				linestream >> c;
				if (c == '"')
				{
					quoteIsOpen = !quoteIsOpen;
				}
				else if (c == ',' && !quoteIsOpen)
				{
					cells.back().push_back(cell);
					cell.clear();
					continue;
				}

				if (linestream.good())
				{
					cell.push_back(c);
				}
				
			}
			cells.back().push_back(cell);
		}

		size_t rows = (props.ignoreHeader) ? cells.size() - 1 : cells.size();
		size_t startRow = (props.ignoreHeader) ? 1 : 0;
		size_t cols = cells.front().size();
		nd::array<T> mat({ rows, cols - columnsToExclude.size() });

		size_t matCol = 0;
		for (size_t c = 0; c < cols; ++c)
		{
			if (columnsToExclude.contains(c)) { continue; }

			size_t matRow = 0;
			for (size_t r = startRow; r < rows; ++r)
			{
				mat({ matRow, matCol }) = columnParser(cells[0][c], cells[r][c]);
				matRow++;
			}

			matCol++;
		}

		return mat;
	}
}