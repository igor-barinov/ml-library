#include "pch.h"

using namespace nd;

template <typename T>
void fill_array(nd::array<T>& arr)
{
	T count = 1;
	for (auto& v : arr)
	{
		v = count;
		count++;
	}
}


TEST(NDArrayTest, TestProps)
{
	nd::array<> empty;
	auto& s1 = empty.shape();
	
	ASSERT_EQ(empty.dims(), 0);
	ASSERT_TRUE(s1.empty());
	ASSERT_EQ(empty.N(), 0);
	ASSERT_TRUE(empty.empty());
	ASSERT_FALSE(empty.matrix());
	ASSERT_FALSE(empty.square());

	nd::array<> mat2d({ 10, 10 });
	auto& s2 = mat2d.shape();
	
	ASSERT_EQ(mat2d.dims(), 2);
	ASSERT_EQ(s2[0], 10);
	ASSERT_EQ(s2[1], 10);
	ASSERT_EQ(mat2d.N(), 100);
	ASSERT_FALSE(mat2d.empty());
	ASSERT_TRUE(mat2d.matrix());
	ASSERT_TRUE(mat2d.square());

	nd::array<> mat3d({ 3, 4, 5 });
	auto& s3 = mat3d.shape();

	ASSERT_EQ(mat3d.dims(), 3);
	ASSERT_EQ(s3[0], 3);
	ASSERT_EQ(s3[1], 4);
	ASSERT_EQ(s3[2], 5);
	ASSERT_EQ(mat3d.N(), 60);
	ASSERT_FALSE(mat3d.empty());
	ASSERT_FALSE(mat3d.matrix());
	ASSERT_FALSE(mat3d.square());
}

TEST(NDArrayTest, TestItemAccess)
{
	nd::array<> empty;
	ASSERT_ANY_THROW(empty.at({ 0 }));

	nd::array<int> mat2d({ 2, 2 });
	fill_array(mat2d);

	/*
	* 1 3
	* 2 4
	*/

	ASSERT_EQ(mat2d({ 0, 0 }), 1);
	ASSERT_EQ(mat2d({ 0, 1 }), 3);
	ASSERT_EQ(mat2d({ 1, 0 }), 2);
	ASSERT_EQ(mat2d({ 1, 1 }), 4);
	ASSERT_ANY_THROW(mat2d.at({ 2, 0 }));
	ASSERT_ANY_THROW(mat2d.at({ 0, 2 }));

	/*
	* 1 4
	* 2 5
	*/
	mat2d({ 0, 1 }) = 4;
	mat2d({ 1, 1 }) = 5;
	ASSERT_EQ(mat2d({ 0, 0 }), 1);
	ASSERT_EQ(mat2d({ 0, 1 }), 4);
	ASSERT_EQ(mat2d({ 1, 0 }), 2);
	ASSERT_EQ(mat2d({ 1, 1 }), 5);


	nd::array<int> mat3d({ 3, 2, 2 });
	fill_array(mat3d);

	/*
	* 1 4	7 10
	* 2 5	8 11
	* 3 6	9 12
	*/

	ASSERT_EQ(mat3d({ 0, 0, 0 }), 1);
	ASSERT_EQ(mat3d({ 0, 1, 0 }), 4);
	ASSERT_EQ(mat3d({ 1, 0, 0 }), 2);
	ASSERT_EQ(mat3d({ 1, 1, 0 }), 5);
	ASSERT_EQ(mat3d({ 2, 0, 0 }), 3);
	ASSERT_EQ(mat3d({ 2, 1, 0 }), 6);
	ASSERT_EQ(mat3d({ 0, 0, 1 }), 7);
	ASSERT_EQ(mat3d({ 0, 1, 1 }), 10);
	ASSERT_EQ(mat3d({ 1, 0, 1 }), 8);
	ASSERT_EQ(mat3d({ 1, 1, 1 }), 11);
	ASSERT_EQ(mat3d({ 2, 0, 1 }), 9);
	ASSERT_EQ(mat3d({ 2, 1, 1 }), 12);
	ASSERT_ANY_THROW(mat3d.at({ 3, 0, 0 }));
	ASSERT_ANY_THROW(mat3d.at({ 0, 2, 0 }));
	ASSERT_ANY_THROW(mat3d.at({ 0, 0, 2 }));
}

TEST(NDArrayTest, TestSlicing)
{
	nd::array<> empty;
	ASSERT_ANY_THROW(empty.slice({1_r}));

	/*
	* 1 4
	* 2 5
	* 3 6
	*/
	nd::array<int> mat2d({ 3, 2 });
	fill_array(mat2d);

	/*
	* 1
	* 2
	*/
	auto slice1 = mat2d({
		2_r,
		1_r
		});

	ASSERT_EQ(slice1.dims(), 2);
	ASSERT_EQ(slice1.N(), 2);
	auto shape1 = slice1.shape();
	ASSERT_EQ(shape1[0], 2);
	ASSERT_EQ(shape1[1], 1);

	ASSERT_EQ(slice1({ 0, 0 }), 1);
	ASSERT_EQ(slice1({ 1, 0 }), 2);

	/*
	* 1 4
	* 3 6
	*/
	auto slice2 = mat2d({
		range(0, 3, 2),
		2_r,
		});

	ASSERT_EQ(slice2.dims(), 2);
	ASSERT_EQ(slice2.N(), 4);
	auto shape2 = slice2.shape();
	ASSERT_EQ(shape2[0], 2);
	ASSERT_EQ(shape2[1], 2);

	ASSERT_EQ(slice2({ 0, 0 }), 1);
	ASSERT_EQ(slice2({ 0, 1 }), 4);
	ASSERT_EQ(slice2({ 1, 0 }), 3);
	ASSERT_EQ(slice2({ 1, 1 }), 6);

	/*
	* 1 4 7		10 13 16
	* 2 5 8		11 14 17
	* 3 6 9		12 15 18
	*/
	nd::array<int> mat3d({ 3, 3, 2 });
	fill_array(mat3d);

	/*
	* 2 5 8		11 14 17
	*/
	auto slice3 = mat3d({
		range(1, 2),
		3_r,
		2_r
		});

	ASSERT_EQ(slice3.dims(), 3);
	ASSERT_EQ(slice3.N(), 6);
	auto shape3 = slice3.shape();
	ASSERT_EQ(shape3[0], 1);
	ASSERT_EQ(shape3[1], 3);
	ASSERT_EQ(shape3[2], 2);

	ASSERT_EQ(slice3({ 0, 0, 0 }), 2);
	ASSERT_EQ(slice3({ 0, 1, 0 }), 5);
	ASSERT_EQ(slice3({ 0, 2, 0 }), 8);
	ASSERT_EQ(slice3({ 0, 0, 1 }), 11);
	ASSERT_EQ(slice3({ 0, 1, 1 }), 14);
	ASSERT_EQ(slice3({ 0, 2, 1 }), 17);
}

TEST(NDArrayTest, TestReshape)
{
	/*
	* 1 3
	* 2 4
	*/
	nd::array<int> mat2d({ 2, 2 });
	fill_array(mat2d);

	ASSERT_ANY_THROW(mat2d.reshape({ 3, 2 }));

	/*
	* 1
	* 2
	* 3
	* 4
	*/
	mat2d.reshape({ 4, 1 });

	ASSERT_EQ(mat2d.dims(), 2);
	ASSERT_EQ(mat2d.N(), 4);
	auto shape1 = mat2d.shape();
	ASSERT_EQ(shape1[0], 4);
	ASSERT_EQ(shape1[1], 1);

	ASSERT_EQ(mat2d({ 0, 0 }), 1);
	ASSERT_EQ(mat2d({ 1, 0 }), 2);
	ASSERT_EQ(mat2d({ 2, 0 }), 3);
	ASSERT_EQ(mat2d({ 3, 0 }), 4);
	ASSERT_ANY_THROW(mat2d.at({ 0, 1 }));
}

TEST(NDArrayTest, TestTranspose)
{
	/*
	* 1 4
	* 2 5
	* 3 6
	*/
	nd::array<int> mat2d({ 3, 2 });
	fill_array(mat2d);

	/*
	* 1 2 3
	* 4 5 6
	*/
	auto T = mat2d.T();

	ASSERT_EQ(T.dims(), 2);
	ASSERT_EQ(T.N(), 6);
	auto shape1 = T.shape();
	ASSERT_EQ(shape1[0], 2);
	ASSERT_EQ(shape1[1], 3);

	ASSERT_EQ(T({ 0, 0 }), 1);
	ASSERT_EQ(T({ 0, 1 }), 2);
	ASSERT_EQ(T({ 0, 2 }), 3);
	ASSERT_EQ(T({ 1, 0 }), 4);
	ASSERT_EQ(T({ 1, 1 }), 5);
	ASSERT_EQ(T({ 1, 2 }), 6);

	nd::array<> mat3d({ 1, 3, 2 });
	ASSERT_ANY_THROW(mat3d.T());
}

TEST(NDArrayTest, TestArithmetic)
{
	/*
	* 1 3
	* 2 4
	*/
	nd::array<int> mat2d({ 2, 2 });
	fill_array(mat2d);

	/*
	* 2 6
	* 4 8
	*/
	auto A = mat2d + mat2d;
	ASSERT_EQ(A.dims(), 2);
	ASSERT_EQ(A.N(), 4);
	ASSERT_EQ(A({ 0, 0 }), 2);
	ASSERT_EQ(A({ 0, 1 }), 6);
	ASSERT_EQ(A({ 1, 0 }), 4);
	ASSERT_EQ(A({ 1, 1 }), 8);

	/*
	* 1 3
	* 2 4
	*/
	auto B = A - mat2d;
	ASSERT_EQ(B.dims(), 2);
	ASSERT_EQ(B.N(), 4);
	ASSERT_EQ(B({ 0, 0 }), 1);
	ASSERT_EQ(B({ 0, 1 }), 3);
	ASSERT_EQ(B({ 1, 0 }), 2);
	ASSERT_EQ(B({ 1, 1 }), 4);

	/*
	* 1 9
	* 4 16
	*/
	auto C = mat2d.hadamard(mat2d);
	ASSERT_EQ(C.dims(), 2);
	ASSERT_EQ(C.N(), 4);
	ASSERT_EQ(C({ 0, 0 }), 1);
	ASSERT_EQ(C({ 0, 1 }), 9);
	ASSERT_EQ(C({ 1, 0 }), 4);
	ASSERT_EQ(C({ 1, 1 }), 16);
}

TEST(NDArrayTest, TestMatMul)
{
	/*
	* 1 4 7
	* 2 5 8
	* 3 6 9
	*/
	nd::array<> mat2d({ 3, 3 });
	fill_array(mat2d);

	/*
	* 1 0 0
	* 0 1 0
	* 0 0 1
	*/
	auto I = nd::array<>::identity(3);

	auto A = mat2d * I;
	ASSERT_TRUE(A.approx_equal(mat2d, 1.0E-10));

	/*
	* 1 3 5 7
	* 2 4 6 8
	*/
	nd::array<> X({ 2, 4 });
	fill_array(X);

	/*
	* 1
	* 2
	* 3
	* 4
	*/
	nd::array<> y({ 4, 1 });
	fill_array(y);

	/*
	* 50
	* 60
	*/
	auto B = X * y;

	ASSERT_EQ(B.dims(), 2);
	auto shape1 = B.shape();
	ASSERT_EQ(shape1[0], 2);
	ASSERT_EQ(shape1[1], 1);


	ASSERT_DOUBLE_EQ(B({ 0, 0 }), 50.0);
	ASSERT_DOUBLE_EQ(B({ 1, 0 }), 60.0);

	ASSERT_ANY_THROW(y * X);
}

TEST(NDArrayTest, TestMean)
{
	/*
	* 1 2	5 6
	* 3 4	7 8
	*/
	nd::array<> mat2d({ 2, 2, 2 });
	fill_array(mat2d);


	auto mean1 = mat2d.mean();
	ASSERT_DOUBLE_EQ(mean1, 4.5);

	/*
	* d=0:
	*
	* 1.5	5.5
	* 3.5	7.5
	*/
	auto mean2 = mat2d.mean(0);

	ASSERT_EQ(mean2.dims(), 3);
	ASSERT_EQ(mean2.N(), 4);
	auto shape1 = mean2.shape();
	ASSERT_EQ(shape1[0], 1);
	ASSERT_EQ(shape1[1], 2);
	ASSERT_EQ(shape1[2], 2);

	ASSERT_DOUBLE_EQ(mean2({ 0, 0, 0 }), 1.5);
	ASSERT_DOUBLE_EQ(mean2({ 0, 1, 0 }), 3.5);
	ASSERT_DOUBLE_EQ(mean2({ 0, 0, 1 }), 5.5);
	ASSERT_DOUBLE_EQ(mean2({ 0, 1, 1 }), 7.5);

	/*
	* d = 1:
	*
	* 2 3  6 7
	*/
	auto mean3 = mat2d.mean(1);

	ASSERT_EQ(mean3.dims(), 3);
	ASSERT_EQ(mean3.N(), 4);
	auto shape2 = mean3.shape();
	ASSERT_EQ(shape2[0], 2);
	ASSERT_EQ(shape2[1], 1);
	ASSERT_EQ(shape2[2], 2);

	ASSERT_DOUBLE_EQ(mean3({ 0, 0, 0 }), 2);
	ASSERT_DOUBLE_EQ(mean3({ 1, 0, 0 }), 3);
	ASSERT_DOUBLE_EQ(mean3({ 0, 0, 1 }), 6);
	ASSERT_DOUBLE_EQ(mean3({ 1, 0, 1 }), 7);

	/*
	* d = 2:
	*
	* 3 4
	* 5 6
	*/
	auto mean4 = mat2d.mean(2);

	ASSERT_EQ(mean4.dims(), 3);
	ASSERT_EQ(mean4.N(), 4);
	auto shape3 = mean4.shape();
	ASSERT_EQ(shape3[0], 2);
	ASSERT_EQ(shape3[1], 2);
	ASSERT_EQ(shape3[2], 1);

	ASSERT_DOUBLE_EQ(mean4({ 0, 0, 0 }), 3);
	ASSERT_DOUBLE_EQ(mean4({ 0, 1, 0 }), 5);
	ASSERT_DOUBLE_EQ(mean4({ 1, 0, 0 }), 4);
	ASSERT_DOUBLE_EQ(mean4({ 1, 1, 0 }), 6);

	ASSERT_ANY_THROW(mat2d.mean(3));
}