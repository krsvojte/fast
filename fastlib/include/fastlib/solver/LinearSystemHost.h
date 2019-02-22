#pragma once

#include <fastlib/solver/LinearSystem.h>
#include <Eigen/Eigen>
#include <Eigen/SparseCore>


namespace fast {

	template <typename T>
	class LinearSystemHost : public LinearSystem {
	public:
		Eigen::SparseMatrix<T, Eigen::RowMajor> A;
		Eigen::Matrix<T, Eigen::Dynamic, 1> x;
		Eigen::Matrix<T, Eigen::Dynamic, 1> b;


		virtual PrimitiveType type() override
		{
			return primitiveTypeof<T>();
		}

	};

}