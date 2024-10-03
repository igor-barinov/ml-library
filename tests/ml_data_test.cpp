#include "pch.h"

using namespace data;

TEST(MLDataTest, TestReadCsv)
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

	csv_props dataProps;	
	dataProps.ignoreHeader = true;
	dataProps.excludedCols = { 4, 14, 15 };
	auto data = read_csv<double>("E:\\Development\\Projects\\ml-library\\data\\ObesityDataSet.csv", dataProps, obesity_data_parser{});

	auto actualMeans = nd::array<>({ 1, 14 });
	actualMeans({ 0, 0 }) = 24.312599908574136;
	actualMeans({ 0, 1 }) = 0.49407863571766936;
	actualMeans({ 0, 2 }) = 1.7016773533870204;
	actualMeans({ 0, 3 }) = 86.58605812648034;
	actualMeans({ 0, 4 }) = 0.8839412600663192;
	actualMeans({ 0, 5 }) = 2.4190430615821885;
	actualMeans({ 0, 6 }) = 2.68562804973946;
	actualMeans({ 0, 7 }) = 0.045476077688299386;
	actualMeans({ 0, 8 }) = 0.020843202273803884;
	actualMeans({ 0, 9 }) = 2.0080114040738986;
	actualMeans({ 0, 10 }) = 0.817621980104216;
	actualMeans({ 0, 11 }) = 1.0102976958787304;
	actualMeans({ 0, 12 }) = 0.657865923732828;
	actualMeans({ 0, 13 }) = 1.8180956892468025;

	auto means = data.mean(0);

	ASSERT_TRUE(means.approx_equal(actualMeans, 0.1));
}