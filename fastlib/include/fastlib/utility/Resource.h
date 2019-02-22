/*
#pragma once

#include <fastlib/FastLibDef.h>
#include "PrimitiveTypes.h"

#include <memory>





namespace fast {

	class ResourceImpl;
	

	//Opaque owning resource of array (1-3D)
	//Must be includable in .cu files
	class Resource {		
	
	
	public:

		enum Location {
			LOCATION_DEVICE = 1,
			LOCATION_HOST = 2
		};

		/ *struct Array1D {
			void * data;
			PrimitiveType type;
			int x;
		};


		struct Array2D {
			void * data;
			PrimitiveType type;
			int x;
			int y;
		};

		struct Array3D {
			void * data;
			PrimitiveType type;
			int x;
			int y;
			int z;
		};* /


		Resource(			 						
			bool onHost = false
		);		

		bool resize(int size);

		/ *
			TODO reduce op
			TODO dot op
			TODO square norm
			TODO copy to GL texture
		* /
		

		//int getDimension();
		//PrimitiveType getType() const;

		/ *Array1D getArray1D(Location loc) const;
		Array2D getArray2D(Location loc) const;
		Array3D getArray3D(Location loc) const;* /


	private:
		//int _dim;
		//PrimitiveType _type;
		//int _x, _y, _z;
				
		std::unique_ptr<ResourceImpl> _impl;
#ifdef UNIFORM_MEMORY_DISABLED
		std::vector<unsigned char> _cpuData;
#endif		

	};


}*/