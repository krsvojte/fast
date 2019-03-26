#define DISTFUN_IMPLEMENTATION
#define DISTFUN_ENABLE_CUDA
#include "distfun/distfun.hpp"

#include "geometry/SDF.h"
#include "volume/Volume.h"
#include "cuda/SDF.cuh"

#include <glm/ext/matrix_transform.hpp>

#include <iostream>
#include <fstream>

namespace fast {

	bool SDFSave(const SDFArray & arr, const std::string & filename)
	{
		std::ofstream f(filename, std::ios::binary);

		if (!f.good()) return false;
		
		//Write number of primitives
		int N = static_cast<int>(arr.size());
		f.write(reinterpret_cast<const char*>(&N), sizeof(int));
		
		struct Grid {
			size_t byteSize;
			size_t index;
		};
		std::unordered_map<const void *, Grid> gridmap;
		std::vector<const void*> gridIndices;

		//Save unique grid index and bytesize
		for (auto & e : arr) {
			if (e.type == distfun::sdPrimitive::SD_GRID) {
				auto it = gridmap.find(e.params.grid.ptr);
				if (it == gridmap.end()) {
					gridIndices.push_back(e.params.grid.ptr);
					size_t byteSize = e.params.grid.size.x * e.params.grid.size.y * e.params.grid.size.z * sizeof(float);
					gridmap[e.params.grid.ptr] = { byteSize, gridIndices.size() - 1 };
				}
			}
		}

		//Write indiviudal primitives
		for (auto & e : arr) {
			distfun::sdPrimitive tmp = e;			
			
			//If its a grid, adjust its pointer to unique index
			if (e.type == distfun::sdPrimitive::SD_GRID) {
				tmp.params.grid.ptr = ((const char*)nullptr) + gridmap[e.params.grid.ptr].index;
			}
			f.write(reinterpret_cast<const char*>(&tmp), sizeof(distfun::sdPrimitive));

		}

		//Write number of grids
		int gridN = static_cast<int>(gridIndices.size());
		f.write(reinterpret_cast<const char*>(&gridN), sizeof(int));

		//Write individual grids
		for (auto ptr : gridIndices) {
			size_t byteSize = gridmap[ptr].byteSize;
			//Write number of bytes that the grid takes
			f.write(reinterpret_cast<const char*>(&byteSize), sizeof(size_t));
			//Write the grid
			f.write(reinterpret_cast<const char*>(ptr), byteSize);			
		}


		return true;
	}
	

	fast::SDFArray SDFLoad(const std::string & filename, std::function<const void *(const distfun::sdGridParam &tempGrid)> gridCallback)
	{

		std::ifstream f(filename, std::ios::binary);

		SDFArray result;
		if (!f.good()) return result;

		int N = 0;
		f.read(reinterpret_cast<char*>(&N), sizeof(int));
		result.resize(N);

		std::unordered_map<const void *, distfun::sdGridParam> gridParams;

		for (auto i = 0; i < N; i++) {
			f.read(reinterpret_cast<char*>(&result[i]), sizeof(distfun::sdPrimitive));		

			//Save grid params
			if (result[i].type == distfun::sdPrimitive::SD_GRID) {
				gridParams[result[i].params.grid.ptr] = result[i].params.grid;
			}
		}

		int gridN = 0;
		f.read(reinterpret_cast<char*>(&gridN), sizeof(int));
		std::vector<char> gridBuffer;

		std::unordered_map<const void *, const void *> newGridPointers;

		for (auto i = 0; i < gridN; i++) {
			size_t byteSize = 0;
			f.read(reinterpret_cast<char*>(&byteSize), sizeof(size_t));

			gridBuffer.resize(byteSize);
			f.read(gridBuffer.data(), byteSize);


			//Callback func that should return newly allocated grid managed by the application
			const void * persistentPointer = ((const char*)nullptr) + i;			
			auto tmpGridParams = gridParams[persistentPointer];
			tmpGridParams.ptr = gridBuffer.data();
			newGridPointers[persistentPointer] = gridCallback(tmpGridParams);
		}

		//Change grid pointers
		for (auto & e : result) {
			if (e.type == distfun::sdPrimitive::SD_GRID) {
				e.params.grid.ptr = newGridPointers[e.params.grid.ptr];
			}
		}

		return result;
	}

	float volumeInBounds(const distfun::sdAABB & bounds, distfun::sdProgramStatic * programDataPtr, int maxDepth) {
		return distfun::sdIntegrateProgramRecursiveExplicit<8, float>(
			programDataPtr,
			bounds,
			distfun::sdIntegrateVolume,
			maxDepth
			);
	}


	SDFArray filterSDFByAABB(const SDFArray & arr, const distfun::sdAABB & domain)
	{

		SDFArray newArr;
		for (auto & p : arr) {
			auto pb = distfun::sdPrimitiveBounds(p, glm::length(domain.diagonal()) * 0.5f);
			if (pb.intersect(domain).isValid()) {
				newArr.push_back(p);
			}
		}

		return newArr;
	}

	distfun::sdProgram SDFToProgram(const SDFArray & state, const distfun::sdAABB * inversionDomain, const distfun::sdPrimitive * intersectionPrimitive)
	{
		if (state.size() == 0) {
			return distfun::sdProgram();
		}
		//Build tree out of _primitives with _sa state transforms 
		auto root = std::make_unique<distfun::sdTreeNode>();
		root->primitive = state.front();

		for (auto i = 1; i < state.size(); i++) {
			auto newNode = std::make_unique<distfun::sdTreeNode>();
			newNode->primitive = state[i];

			auto newRoot = std::make_unique<distfun::sdTreeNode>();
			newRoot->primitive.type = distfun::sdPrimitive::Type::SD_OP_UNION;
			newRoot->children[0] = std::move(root);
			newRoot->children[1] = std::move(newNode);
			root = std::move(newRoot);
		}

		if (inversionDomain) {
			distfun::sdPrimitive box;
			box.params.box.size = inversionDomain->diagonal() * 0.5f;
			box.invTransform = glm::inverse(glm::translate(mat4(1.0f), inversionDomain->center()));
			box.type = distfun::sdPrimitive::SD_BOX;
			box.rounding = 0.0f;

			auto newNode = std::make_unique<distfun::sdTreeNode>();
			newNode->primitive = box;

			auto newRoot = std::make_unique<distfun::sdTreeNode>();
			newRoot->primitive.type = distfun::sdPrimitive::Type::SD_OP_DIFFERENCE;
			newRoot->children[0] = std::move(root);
			newRoot->children[1] = std::move(newNode);
			root = std::move(newRoot);			
		}

		if (intersectionPrimitive) {

			auto newNode = std::make_unique<distfun::sdTreeNode>();
			newNode->primitive = *intersectionPrimitive;

			auto newRoot = std::make_unique<distfun::sdTreeNode>();
			newRoot->primitive.type = distfun::sdPrimitive::Type::SD_OP_INTERSECT;
			newRoot->children[0] = std::move(root);
			newRoot->children[1] = std::move(newNode);
			root = std::move(newRoot);
		
		}

		return distfun::sdCompile(*root);
	}



	distfun::sdProgram SDFOverlapProgram(
		const SDFArray & arr,
		const distfun::sdAABB & domain
	){



		distfun::sdProgram p;

		if (arr.size() == 0) {
			return distfun::sdProgram();
		}

		//Primitive bounds
		std::vector<distfun::sdAABB> aabbs;
		aabbs.resize(arr.size(), distfun::sdAABB());

		for (auto i = 0; i < arr.size(); i++) {
			aabbs[i] = distfun::sdPrimitiveBounds(arr[i], glm::length(domain.diagonal()) * 0.5f);
		}

		std::vector<std::pair<int, int>> pairs;
		std::vector<std::unique_ptr<distfun::sdTreeNode>> subtrees;

		for (auto i = 0; i < arr.size(); i++) {
			for (auto j = i + 1; j < arr.size(); j++) {
				if (!aabbs[i].intersect(aabbs[j]).isValid()) continue;					

				auto nodeI = std::make_unique<distfun::sdTreeNode>();
				nodeI->primitive = arr[i];

				auto nodeJ = std::make_unique<distfun::sdTreeNode>();
				nodeJ->primitive = arr[j];


				auto subRoot = std::make_unique<distfun::sdTreeNode>();
				subRoot->primitive.type = distfun::sdPrimitive::Type::SD_OP_INTERSECT;
				subRoot->children[0] = std::move(nodeI);
				subRoot->children[1] = std::move(nodeJ);

				subtrees.push_back(std::move(subRoot));								
			}
		}

		std::cout << "Collision subtrees: " << arr.size() << std::endl;

		//No overlaps, return empty progra
		if (subtrees.size() == 0) {
			return p;
		}

		//One overlap, return subtree directly
		if(subtrees.size() == 1){
			return distfun::sdCompile(*subtrees.front());
		}

		//Two or more, consturct tree tail
		auto root = std::make_unique<distfun::sdTreeNode>();
		root->primitive.type = distfun::sdPrimitive::SD_OP_UNION;
		root->children[0] = std::move(subtrees[0]);
		root->children[1] = std::move(subtrees[1]);

		//Add depth to tree
		for (auto i = 2; i < subtrees.size(); i++) {
			auto newRoot = std::make_unique<distfun::sdTreeNode>();
			newRoot->primitive.type = distfun::sdPrimitive::SD_OP_UNION;
			newRoot->children[0] = std::move(root);
			newRoot->children[1] = std::move(subtrees[i]);
			root = std::move(newRoot);			
		}

		return distfun::sdCompile(*root);
	}

	void SDFRasterize(
		const SDFArray & arr, 
		const distfun::sdAABB & domain, 
		Volume & volume,
		bool invert,
		bool commitToGPU,
		bool overlap
	)
	{

		std::vector<unsigned char> programData;
		
		if (overlap) {
			auto program = SDFOverlapProgram(arr, domain);
			programData.resize(program.staticSize());
			
			distfun::sdCommitCPU(programData.data(), program);
		}
		else {
			auto program = SDFToProgram(arr, invert ? &domain : nullptr);
			programData.resize(program.staticSize());
			distfun::sdCommitCPU(programData.data(), program);
		}	

		distfun::sdProgramStatic *programDataPtr = 
			reinterpret_cast<distfun::sdProgramStatic*>(programData.data());


		uchar * volptr = (uchar*)volume.getPtr().getCPU();
		

		vec3 cellSize = vec3(
			domain.diagonal().x / volume.dim().x, 
			domain.diagonal().y / volume.dim().y, 
			domain.diagonal().z / volume.dim().z
		);

		float cellVolume = cellSize.x * cellSize.y * cellSize.z;

		if (overlap) {

			float minCellSize = glm::min(cellSize.x, glm::min(cellSize.y, cellSize.z));

#ifndef _DEBUG
			#pragma omp parallel for			
#endif
			for (auto z = 0; z < volume.dim().z; z++) {
				for (auto y = 0; y < volume.dim().y; y++) {
					for (auto x = 0; x < volume.dim().x; x++) {
						auto index = linearIndex(volume.dim(), ivec3(x, y, z));
						distfun::vec3 pos = domain.min + vec3(cellSize.x * x, cellSize.y * y, cellSize.z* z) + cellSize * 0.5f;
						float d = distfun::sdDistanceAtPos<64>(pos, programDataPtr);
						if (d >= 0.0f) {
							volptr[index] = 0;
							continue;
						}

						distfun::sdAABB sampleBB = { pos - cellSize * 0.5f, pos + cellSize * 0.5f };
						int sampleN = 2;
						vec3 nsum = vec3(0);
						for (auto ix = 0; ix < sampleN; ix++) {
							for (auto iy = 0; iy < sampleN; iy++) {
								for (auto iz = 0; iz < sampleN; iz++) {
									vec3 spos = sampleBB.getSubGrid(ivec3(2), ivec3(ix,iy,iz)).center();
									float eps = minCellSize * 1;// / 10.0f;
									vec3 n = distfun::sdNormal(spos, eps, distfun::sdDistanceAtPos<64>, programDataPtr);
									nsum += n;
									/*float nmag = glm::length(n);
									if (nmag < eps*2.0f)
										volptr[index] = 255;
									else
										volptr[index] = 0;*/
								}
							}
						}

						nsum = nsum * (1.0f / (sampleN*sampleN*sampleN));
						if (glm::length(nsum) < 0.990f){
							volptr[index] = 255;
						}
						else {
							volptr[index] = 0;
						}
						

											
					}
				}
			}
		
		
		}
		else {
			#pragma omp parallel for			
			for (auto z = 0; z < volume.dim().z; z++) {
				for (auto y = 0; y < volume.dim().y; y++) {
					for (auto x = 0; x < volume.dim().x; x++) {
						auto index = linearIndex(volume.dim(), ivec3(x, y, z));
						distfun::vec3 pos = domain.min + vec3(cellSize.x * x, cellSize.y * y, cellSize.z* z) + cellSize * 0.5f;
						float d = distfun::sdDistanceAtPos<64>(pos, programDataPtr);
						volptr[index] = (d < 0.0f) ? 255 : 0;
					}
				}
			}
		}
		


		if (commitToGPU) {
			volume.getPtr().commit();
			cudaDeviceSynchronize();
		}
	}





	float SDFVolume(
		const SDFArray & arr,
		const distfun::sdAABB & domain,
		int maxDepth, /* = 4*/
		bool onDevice /*= false */
	)
	{

		if (!onDevice) {
			float actualVolume = 0.0f;
			ivec3 grid = ivec3(4);
#pragma omp parallel for
			for (auto z = 0; z < grid.z; z++) {
				for (auto y = 0; y < grid.y; y++) {
					for (auto x = 0; x < grid.x; x++) {

						auto subgrid = domain.getSubGrid(grid, { x,y,z });
						auto program = SDFToProgram(filterSDFByAABB(arr, subgrid));
						if (program.instructions.size() == 0)
							continue;

						std::vector<unsigned char> programData(program.staticSize());
						distfun::sdCommitCPU(programData.data(), program);
						distfun::sdProgramStatic *programDataPtr = reinterpret_cast<distfun::sdProgramStatic*>(programData.data());

						float val = volumeInBounds(subgrid, programDataPtr, maxDepth);

						/*float val = distfun::volumeInBounds(
							subgrid,
							programDataPtr,
							0, maxDepth
						);*/

#pragma omp atomic						
						actualVolume += val;
					}

				}
			}

			return actualVolume;
		}


		
		//On device
		assert(false);
		//TODO

	}


	vec4 voidGravity(
		vec3 pt0,
		float m0,
		const distfun::sdAABB & bounds,
		distfun::sdProgramStatic * program,		
		int curDepth,
		int maxDepth
	) {
		const vec3 pt = bounds.center();
		const float d = distfun::sdDistanceAtPos<8>(pt, program);

		//If nearest surface is outside of bounds
		const vec3 diagonal = bounds.diagonal();
		if (curDepth == maxDepth || d*d >= 0.5f * 0.5f * glm::length2(diagonal)) {
			//Cell completely outside
			if (d > 0.0f) return vec4(0.0f);

			//Cell completely inside			

			float mass = bounds.volume();
			
			float G = 9.81f;
			vec3 dir = pt - pt0;
			float magnitude = (mass * m0) / glm::length2(pt0 - pt);
			vec3 F = glm::normalize(dir) * magnitude;

			return vec4(F, magnitude);

			//Distance to nearest non-penetrating surface of a
			//const float L = distDifference(da, db);

			//Normal to nearest non-penetrating surface of a
			/*const vec3 N = sdNormal(pt, 0.0001f, distPrimitiveDifference, a, b);
			const float magnitude = 0.5f * k * (L*L);
			const vec3 U = magnitude * N;
			return vec4(U, magnitude);*/
		}

		//Nearest surface is within bounds, subdivide
		vec4 x = vec4(0.0f);		
		for (auto i = 0; i < 8; i++) {
			x += voidGravity(pt0,m0, bounds.getOctant(i), program, curDepth + 1, maxDepth);
		}

		return x;



	}

	std::vector<vec4> SDFElasticity(const SDFArray & s, const distfun::sdAABB & domain, RNGUniformFloat & rng, int maxDepth /*= 3*/, bool onDevice /*= false */)
	{

		std::vector<vec4> result(s.size(),vec4(0.0f));


		if (onDevice) {
			//On device
			assert(false);
			//TODO
			return result;
		}


		//Primitive bounds
		std::vector<distfun::sdAABB> aabbs;
		aabbs.resize(s.size(), distfun::sdAABB());

		for (auto i = 0; i < s.size(); i++) {
			aabbs[i] = distfun::sdPrimitiveBounds(s[i], glm::length(domain.diagonal()) * 0.5f);
		}


		
	
		
		#pragma omp parallel for
		for (auto i = 0; i < s.size(); i++) {
			auto pt = aabbs[i].center();

			auto & bb = aabbs[i];
			
			auto program = SDFToProgram(filterSDFByAABB(s, bb), &domain);
			std::vector<unsigned char> programData(program.staticSize());
			distfun::sdCommitCPU(programData.data(), program);
			distfun::sdProgramStatic *programDataPtr = reinterpret_cast<distfun::sdProgramStatic*>(programData.data());

			
			//result[i] = voidGravity(pt, domain, programDataPtr, 0, maxDepth);
			
			
			float m0 = 1.0f;
			result[i] = voidGravity(pt,m0, bb, programDataPtr, 0, maxDepth);
			//result[i].w *= -1.0f;
			//result[i].w = float(objCol[i]);

			

		}

		
		//return result;
		

		

		//Far collision phase
		std::vector<PrimitiveCollisionPair> pairs;
		for (auto i = 0; i < s.size(); i++) {
			const distfun::sdAABB & abb = aabbs[i];
			for (auto j = 0; j < s.size(); j++) {

				if (i == j) continue;
				const distfun::sdAABB & bbb = aabbs[j];
				const distfun::sdAABB isect = abb.intersect(bbb);
				if (!isect.isValid())
					continue;

				pairs.push_back({i, j, isect});
			}
		}

		//std::cout << "Collisions: " << pairs.size() << std::endl;

		
		
		
		float totalVolume = domain.volume();

		#pragma omp parallel for
		for (auto pi = 0; pi < pairs.size(); pi++) {
			auto & pair = pairs[pi];
			auto & a = s[pair.indexA];
			auto & b = s[pair.indexB];
			auto & abb = aabbs[pair.indexA];
			auto & bbb = aabbs[pair.indexB];
			auto & isect = pair.bounds;

			float ratioA = isect.volume() / abb.volume();
			float ratioB = isect.volume() / bbb.volume();
			if (ratioA > 0.90f && ratioB > 0.90f) {
				vec4 Um = (vec4(rng.next(), rng.next(), rng.next(), 1.0f) * 2.0f - vec4(1.0f))
					* glm::length(abb.diagonal());
				//#pragma omp atomic
				#pragma omp atomic			
				result[pair.indexA].x += Um.x;
				#pragma omp atomic
				result[pair.indexA].y += Um.y;
				#pragma omp atomic
				result[pair.indexA].z += Um.z;
				#pragma omp atomic
				result[pair.indexA].w += Um.w;

				continue;
			}

			vec3 diff = (abb.center() - bbb.center());
			float dist = glm::length(diff);

			float eps = 0.0001f;
			vec3 N;
			if (dist < eps) {
				dist = eps;
				N = glm::normalize(vec3(rng.next(), rng.next(), rng.next()));
			}
			else {
				N = glm::normalize(diff);
			}			
			//float dx = glm::length(diff) / glm::length(abb.center() + bbb.center()) ;					
			

			float dx = 1.0f / (dist*dist);

			vec3 F = (N * dx);// / float(s.size());

			
			assert(!std::isnan(F.x));

			F *= 0.05f * abb.volume();
			vec4 Um = vec4(F, glm::length(F));


			/*float k = (totalVolume / isect.volume()) * (8.0f / 1.0f);			

			vec4 Um = distfun::sdPrimitiveElasticity(isect, a, b, k, 0, maxDepth);*/
			
			#pragma omp atomic			
			result[pair.indexA].x += Um.x;
			#pragma omp atomic
			result[pair.indexA].y += Um.y;
			#pragma omp atomic
			result[pair.indexA].z += Um.z;
			#pragma omp atomic
			result[pair.indexA].w += Um.w;
			
		}

		return result;


		


		





	}

	std::vector<float> SDFPerParticleOverlap(const SDFArray & s, const distfun::sdAABB & domain, int maxDepth /*= 0 */)
	{

		std::vector<float> res(s.size(),0.0f);

		//Primitive bounds
		std::vector<distfun::sdAABB> aabbs;
		aabbs.resize(s.size(), distfun::sdAABB());

		for (auto i = 0; i < s.size(); i++) {
			aabbs[i] = distfun::sdPrimitiveBounds(s[i], glm::length(domain.diagonal()) * 0.5f);
		}


		//std::vector<PrimitiveCollisionPair> pairs;
		//std::vector<int> objCol(s.size(), 0);
		std::vector<std::vector<distfun::sdPrimitive>> collisions;
		collisions.resize(s.size());

		for (auto i = 0; i < s.size(); i++) {
			const distfun::sdAABB & abb = aabbs[i];			
			for (auto j = i + 1; j < s.size(); j++) {

				//if (i == j) continue;
				const distfun::sdAABB & bbb = aabbs[j];
				const distfun::sdAABB isect = abb.intersect(bbb);
				if (!isect.isValid())
					continue;

				float v = distfun::sdIntersectionVolume(isect, s[i], s[j], 0, maxDepth);
				res[i] += v;
				res[j] += v;
					

				//collisions[i].push_back(s[j]);
				//collisions[j].push_back(s[j]);
				/*objCol[i]++;
				objCol[j]++;*/
				//pairs.push_back({ i, j, isect });
			}
		}

		return res;
		
		//#pragma omp parallel for

		for (auto i = 0; i < s.size(); i++) {
			auto & cols = collisions[i];
			if (cols.size() == 0) continue;

			auto program = SDFToProgram(cols, nullptr, &s[i]);
			std::vector<unsigned char> programData(program.staticSize());
			distfun::sdCommitCPU(programData.data(), program);
			distfun::sdProgramStatic *programDataPtr = reinterpret_cast<distfun::sdProgramStatic*>(programData.data());
			
			float val = volumeInBounds(aabbs[i], programDataPtr, maxDepth);
			res[i] = val;
		}

		return res;


	}

}