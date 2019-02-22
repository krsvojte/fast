#include "geometry/SDF.h"
#include "volume/Volume.h"
#include "cuda/SDF.cuh"

#include <glm/ext/matrix_transform.hpp>

#include <iostream>

namespace fast {

	SDFArray filterSDFByAABB(const SDFArray & arr, const distfun::AABB & domain)
	{

		SDFArray newArr;
		for (auto & p : arr) {
			auto pb = primitiveBounds(p, glm::length(domain.diagonal()) * 0.5f);
			if (pb.intersect(domain).isValid()) {
				newArr.push_back(p);
			}
		}

		return newArr;
	}

	distfun::DistProgram SDFToProgram(const SDFArray & state, const distfun::AABB * inversionDomain, const distfun::Primitive * intersectionPrimitive)
	{
		if (state.size() == 0) {
			return distfun::DistProgram();
		}
		//Build tree out of _primitives with _sa state transforms 
		auto root = std::make_unique<distfun::TreeNode>();
		root->primitive = state.front();

		for (auto i = 1; i < state.size(); i++) {
			auto newNode = std::make_unique<distfun::TreeNode>();
			newNode->primitive = state[i];

			auto newRoot = std::make_unique<distfun::TreeNode>();
			newRoot->primitive.type = distfun::Primitive::Type::SD_OP_UNION;
			newRoot->children[0] = std::move(root);
			newRoot->children[1] = std::move(newNode);
			root = std::move(newRoot);
		}

		if (inversionDomain) {
			distfun::Primitive box;
			box.params.box.size = inversionDomain->diagonal() * 0.5f;
			box.invTransform = glm::inverse(glm::translate(mat4(1.0f), inversionDomain->center()));
			box.type = distfun::Primitive::SD_BOX;
			box.rounding = 0.0f;

			auto newNode = std::make_unique<distfun::TreeNode>();
			newNode->primitive = box;

			auto newRoot = std::make_unique<distfun::TreeNode>();
			newRoot->primitive.type = distfun::Primitive::Type::SD_OP_DIFFERENCE;
			newRoot->children[0] = std::move(root);
			newRoot->children[1] = std::move(newNode);
			root = std::move(newRoot);			
		}

		if (intersectionPrimitive) {

			auto newNode = std::make_unique<distfun::TreeNode>();
			newNode->primitive = *intersectionPrimitive;

			auto newRoot = std::make_unique<distfun::TreeNode>();
			newRoot->primitive.type = distfun::Primitive::Type::SD_OP_INTERSECT;
			newRoot->children[0] = std::move(root);
			newRoot->children[1] = std::move(newNode);
			root = std::move(newRoot);
		
		}

		return distfun::compileProgram(*root);
	}



	void SDFRasterize(
		const SDFArray & arr, 
		const distfun::AABB & domain, 
		Volume & volume,
		bool invert,
		bool commitToGPU
	)
	{

		
		
		auto program = SDFToProgram(arr, invert ? &domain : nullptr);

		std::vector<unsigned char> programData(program.staticSize());
		commitProgramCPU(programData.data(), program);
		distfun::DistProgramStatic *programDataPtr = reinterpret_cast<distfun::DistProgramStatic*>(programData.data());


		uchar * volptr = (uchar*)volume.getPtr().getCPU();
		

		vec3 cellSize = vec3(
			domain.diagonal().x / volume.dim().x, 
			domain.diagonal().y / volume.dim().y, 
			domain.diagonal().z / volume.dim().z
		);

		float cellVolume = cellSize.x * cellSize.y * cellSize.z;

		#pragma omp parallel for			
		for (auto z = 0; z < volume.dim().z; z++) {
			for (auto y = 0; y < volume.dim().y; y++) {
				for (auto x = 0; x <  volume.dim().x; x++) {
					auto index = linearIndex(volume.dim(), ivec3(x, y, z));
					distfun::vec3 pos = domain.min + vec3(cellSize.x * x, cellSize.y * y, cellSize.z* z);
					float d = distfun::distanceAtPos<64>(pos, programDataPtr);
					volptr[index] = (d < 0.0f) ? 255 : 0;
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
		const distfun::AABB & domain,
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
						commitProgramCPU(programData.data(), program);
						distfun::DistProgramStatic *programDataPtr = reinterpret_cast<distfun::DistProgramStatic*>(programData.data());

						float val = distfun::volumeInBounds(
							subgrid,
							programDataPtr,
							0, maxDepth
						);

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
		const distfun::AABB & bounds,
		distfun::DistProgramStatic * program,		
		int curDepth,
		int maxDepth
	) {
		const vec3 pt = bounds.center();
		const float d = distfun::distanceAtPos<8>(pt, program);

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
			/*const vec3 N = distNormal(pt, 0.0001f, distPrimitiveDifference, a, b);
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

	std::vector<vec4> SDFElasticity(const SDFArray & s, const distfun::AABB & domain, RNGUniformFloat & rng, int maxDepth /*= 3*/, bool onDevice /*= false */)
	{

		std::vector<vec4> result(s.size(),vec4(0.0f));


		if (onDevice) {
			//On device
			assert(false);
			//TODO
			return result;
		}


		//Primitive bounds
		std::vector<distfun::AABB> aabbs;
		aabbs.resize(s.size(), distfun::AABB());

		for (auto i = 0; i < s.size(); i++) {
			aabbs[i] = primitiveBounds(s[i], glm::length(domain.diagonal()) * 0.5f);
		}


		
	
		
		#pragma omp parallel for
		for (auto i = 0; i < s.size(); i++) {
			auto pt = aabbs[i].center();

			auto & bb = aabbs[i];
			
			auto program = SDFToProgram(filterSDFByAABB(s, bb), &domain);
			std::vector<unsigned char> programData(program.staticSize());
			commitProgramCPU(programData.data(), program);
			distfun::DistProgramStatic *programDataPtr = reinterpret_cast<distfun::DistProgramStatic*>(programData.data());

			
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
			const distfun::AABB & abb = aabbs[i];
			for (auto j = 0; j < s.size(); j++) {

				if (i == j) continue;
				const distfun::AABB & bbb = aabbs[j];
				const distfun::AABB isect = abb.intersect(bbb);
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

			vec4 Um = distfun::primitiveElasticity(isect, a, b, k, 0, maxDepth);*/
			
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

	std::vector<float> SDFPerParticleOverlap(const SDFArray & s, const distfun::AABB & domain, int maxDepth /*= 0 */)
	{

		std::vector<float> res(s.size(),0.0f);

		//Primitive bounds
		std::vector<distfun::AABB> aabbs;
		aabbs.resize(s.size(), distfun::AABB());

		for (auto i = 0; i < s.size(); i++) {
			aabbs[i] = primitiveBounds(s[i], glm::length(domain.diagonal()) * 0.5f);
		}


		//std::vector<PrimitiveCollisionPair> pairs;
		//std::vector<int> objCol(s.size(), 0);
		std::vector<std::vector<distfun::Primitive>> collisions;
		collisions.resize(s.size());

		for (auto i = 0; i < s.size(); i++) {
			const distfun::AABB & abb = aabbs[i];			
			for (auto j = i + 1; j < s.size(); j++) {

				//if (i == j) continue;
				const distfun::AABB & bbb = aabbs[j];
				const distfun::AABB isect = abb.intersect(bbb);
				if (!isect.isValid())
					continue;

				float v = intersectionVolume(isect, s[i], s[j], 0, maxDepth);
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
			commitProgramCPU(programData.data(), program);
			distfun::DistProgramStatic *programDataPtr = reinterpret_cast<distfun::DistProgramStatic*>(programData.data());
			
			float val = distfun::volumeInBounds(
				aabbs[i],
				programDataPtr,
				0, maxDepth
			);
			res[i] = val;
		}

		return res;


	}

}