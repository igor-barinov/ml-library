#include "pch.h"

using namespace ml::optimizers;

void load_data(ml::matrix_t& X, ml::matrix_t& y)
{
	struct obesity_data_parser
	{
		double operator()(const std::string& header, const std::string& value)
		{
			auto yes_or_no = [](const std::string& yn)
				{
					if (yn == "no") { return 0.0; }
					return 1.0;
				};

			if (header == "Gender")
			{
				if (value == "Male") { return 0.0; }
				return 1.0;
			}
			else if (header == "FAVC")
			{
				return yes_or_no(value);
			}
			else if (header == "SCC")
			{
				return yes_or_no(value);
			}
			else if (header == "SMOKE")
			{
				return yes_or_no(value);
			}
			else if (header == "family_history_with_overweight")
			{
				return yes_or_no(value);
			}
			else if (header == "NObeyesdad")
			{
				if (value == "Insufficient_Weight") { return -1.0; }
				if (value == "Normal_Weight") { return 0.0; }
				if (value == "Overweight_Level_I") { return 1.0; }
				if (value == "Overweight_Level_II") { return 2.0; }
				if (value == "Obesity_Type_I") { return 3.0; }
				if (value == "Obesity_Type_II") { return 3.0; }
				if (value == "Obesity_Type_III") { return 4.0; }
			}

			return std::stod(value);
		}
	};

	data::csv_props dataProps;
	dataProps.ignoreHeader = true;
	dataProps.excludedCols = { 4, 14, 15 };
	auto data = data::read_csv<double>("E:\\Development\\Projects\\ml-library\\data\\ObesityDataSet.csv", dataProps, obesity_data_parser{});

	X = data({
		nd::range(1000),
		nd::range(0, 13)
		});

	y = data({
		nd::range(1000),
		nd::range(13, 14)
		});
}

TEST(MLOptimizerTest, TestSGD)
{
	ml::matrix_t y;
	ml::matrix_t X;

	load_data(X, y);

	ml::regression::logistic logreg(y, X);

	SGD sgd(ml::metrics::cross_entropy, 0.0001, 100);
	sgd.optimize(logreg, {X}, y);
}