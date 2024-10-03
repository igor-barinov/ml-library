#include "pch.h"

using namespace ml;

TEST(MLNetsTest, TestMLP)
{
	nets::mlp mlp({ 10, 64, 10 });
}