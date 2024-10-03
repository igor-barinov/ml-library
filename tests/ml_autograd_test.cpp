#include "pch.h"

#include <numbers>

using namespace ml::autograd;

ml::matrix_t scalar(double value) { return ml::matrix_t({ 1 }, value); }
ml::matrix_t vector(size_t n) { return ml::ones({ n }); }


class ScalarFunc : public differentiable
{
public:
	parameter operator()(const std::vector<parameter>& params) const
	{
		auto x = params[0];

		return exp(sqrt(sin(2.0 * x)));
	}
};

TEST(MLAutogradTest, TestScalarDerivative)
{
	parameter x(scalar(std::numbers::pi / 4.0));

	auto f = ScalarFunc{}({ x });

	auto ddx = f.partial_wrt(x.id());

	auto sqrtsinx = 2.0 * sqrt(sin(2.0 * x));

	auto trueDeriv = exp(x) * (1.0 / sqrtsinx) * cos(2 * x) * 2;
	auto trueVal = trueDeriv.value();


	ASSERT_TRUE(ddx.approx_equal(trueVal));
}

class VectorFunc : public differentiable
{
public:
	VectorFunc(const parameter& wT)
		: wT(wT)
	{
	}

	parameter wT;

	parameter operator()(const std::vector<parameter>& params) const
	{
		auto v = params[0];

		return sqrt(wT * (2.0 * v)) / 2;
	}
};

TEST(MLAutogradTest, TestVectorDerivative)
{
	parameter wT = vector(3) * 2;
	parameter v = vector(3);

	auto func = VectorFunc(wT);

	auto f = func({ v });

	auto ddv = f.partial_wrt(v.id());

	auto trueDeriv = (1.0 / (2 * sqrt(wT * (2.0 * v)))) * (2.0 * wT) * 0.5;
	auto trueVal = trueDeriv.value();

	ASSERT_TRUE(ddv.approx_equal(trueVal));
}

class MatrixFunc : public differentiable
{

};