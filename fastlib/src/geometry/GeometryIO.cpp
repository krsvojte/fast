#include "geometry/GeometryIO.h"


#include "geometry/GeometryObject.h"
#include "quickhull/QuickHull.hpp"

#include <fstream>
#include <set>
#include <algorithm>

#include <iostream>
#include <sstream>
#include <cstring>


namespace fast {


	FAST_EXPORT TriangleMesh loadParticleMesh(const std::string & path)
	{
		using Edge = TriangleMesh::Edge;
		using Face = TriangleMesh::Face;

		TriangleMesh m;
		

		std::ifstream f(path);

		if (!f.good())
			throw "loadParticleMesh invalid file";		


		int nv, nf;
		f >> nv;

		if (nv <= 0)
			throw "loadParticleMesh invalid number of vertices";

		m.vertices.resize(nv);

		for (auto i = 0; i < nv; i++) {
			f >> m.vertices[i].x >> m.vertices[i].y >> m.vertices[i].z;
		}

		f >> nf;

		if (nf <= 0)
			throw "loadParticleMesh invalid number of faces";

		m.faces.resize(nf);

		auto cmpEdge = [](const Edge & a, const Edge & b) {
			Edge as = a;
			if (as.x > as.y) std::swap(as.x, as.y);
			Edge bs = b;
			if (bs.x > bs.y) std::swap(bs.x, bs.y);

			if (as.x < bs.x)
				return true;
			if (as.x > bs.x)
				return false;
			return (as.y < bs.y);
		};

		std::set<Edge, decltype(cmpEdge)> edgeSet(cmpEdge);


		for (auto i = 0; i < nf; i++) {

			Face & face = m.faces[i];

			//Read number of vertices in this face
			int nfv;
			f >> nfv;

			assert(nfv >= 3);

			//Load vertex indices
			face.vertices.resize(nfv);
			for (auto k = 0; k < nfv; k++) {
				f >> face.vertices[k];
			}

			//Save edges
			for (auto k = 0; k < nfv; k++) {
				Edge e = { face.vertices[k], face.vertices[(k + 1) % nfv] };
				edgeSet.insert(e);
			}
		}
		m.edges.resize(edgeSet.size());
		std::copy(edgeSet.begin(), edgeSet.end(), m.edges.begin());

		m.recomputeNormals();

		return m;
	}

	TriangleMesh halfEdgeMeshToTriangleMesh(const quickhull::HalfEdgeMesh<float, size_t> & hemesh) {

		using namespace quickhull;

		TriangleMesh tm;
		tm.vertices.resize(hemesh.m_vertices.size());
		std::memcpy(tm.vertices.data(), hemesh.m_vertices.data(), hemesh.m_vertices.size() * sizeof(vec3));

		auto cmpEdge = [](const TriangleMesh::Edge & a, const TriangleMesh::Edge & b) {
			TriangleMesh::Edge as = a;
			if (as.x > as.y) std::swap(as.x, as.y);
			TriangleMesh::Edge bs = b;
			if (bs.x > bs.y) std::swap(bs.x, bs.y);

			if (as.x < bs.x)
				return true;
			if (as.x > bs.x)
				return false;
			return (as.y < bs.y);
		};

		std::set<TriangleMesh::Edge, decltype(cmpEdge)> edgeSet(cmpEdge);

		for(auto i=0; i < hemesh.m_faces.size(); i++){


			
			auto heIndexStart = hemesh.m_faces[i].m_halfEdgeIndex;
			auto he = hemesh.m_halfEdges[heIndexStart];
			auto heIndexNext = he.m_next;

			TriangleMesh::Face newF;
			
			newF.vertices.push_back(int(he.m_endVertex));

			while (heIndexNext != heIndexStart) {				
				he = hemesh.m_halfEdges[heIndexNext];
				newF.vertices.push_back(int(he.m_endVertex));
				heIndexNext = he.m_next;
			}

			tm.faces.push_back(newF);
			

			for (auto k = 0; k < newF.vertices.size(); k++) {
				TriangleMesh::Edge e = { newF.vertices[k], newF.vertices[(k + 1) % newF.vertices.size()] };
				edgeSet.insert(e);
			}
				

		
		}

		tm.edges.resize(edgeSet.size());
		std::copy(edgeSet.begin(), edgeSet.end(), tm.edges.begin());
		tm.recomputeNormals();

		return tm;


	
	}

	TriangleMesh hullCoordsToMesh(const std::vector<vec3> & pts) {	
		using namespace quickhull;

		QuickHull<float> qh;
		HalfEdgeMesh<float, size_t> hullMesh = qh.getConvexHullAsMesh(
			reinterpret_cast<const float*>(pts.data()), pts.size(), false
		);

		return halfEdgeMeshToTriangleMesh(hullMesh);

		
	}


	FAST_EXPORT std::vector<std::shared_ptr<GeometryObject>> readPosFile(
		std::ifstream & stream, 
		size_t index,
		AABB trim
		)
	{

		std::vector<std::shared_ptr<GeometryObject>> res;
		#define CMP_MATCH 0

		enum ShapeType {
			SHAPE_SPHERE,
			SHAPE_POLY
		};
		
		const std::string boxS = "boxMatrix";
		const std::string defS = "def";
		const std::string eofS = "eof";
		std::string defName = "";		
		char line[4096];

		ShapeType shapeType;

		AABB bb;
		
		vec3 scale;
		vec3 trimRange = trim.range();
		

		std::vector<vec3> coords;

		std::shared_ptr<Geometry> templateParticle;

		size_t curCount = 0;

		//Seek to index
		while (stream.good()) {
			size_t pos = stream.tellg();
			stream.getline(line, 4096);

			if (boxS.compare(0, boxS.length(), line, 0, boxS.length()) == CMP_MATCH) {
				if (curCount == index) {
					std::stringstream ss;
					ss << (line + boxS.length());

					
					bb.min = vec3(0);
					vec3 v[3];
					ss >> v[0].x >> v[0].y >> v[0].z;
					ss >> v[1].x >> v[1].y >> v[1].z;
					ss >> v[2].x >> v[2].y >> v[2].z;

					bb.max = vec3(v[0].x, v[1].y, v[2].z);

					scale = vec3(1.0f / bb.range().x, 1.0f / bb.range().y, 1.0f / bb.range().z);

					std::cout << line << std::endl;
					std::cout << bb.max.x << ", " << bb.max.y << ", " << bb.max.z << std::endl;

					break;
				}
				curCount++;
			}			

		}


		while (stream.good()) {
			size_t pos = stream.tellg();
			stream.getline(line, 4096);
			//auto s = std::string(line);						
			
			/*if (boxS.compare(0, boxS.length(), line, 0, boxS.length()) == CMP_MATCH) {
				
					
			}*/

			if (eofS.compare(0, eofS.length(), line, 0, eofS.length()) == CMP_MATCH) {
				break;
			}

			if (defS.compare(0, defS.length(), line, 0, defS.length()) == CMP_MATCH) {
				std::stringstream ss;
				ss << (line + defS.length() + 1);
				
				getline(ss, defName, ' ');
				
				std::string tmp;
				getline(ss, tmp, '"');				
				getline(ss, tmp, ' ');
				if (tmp == "poly3d")
					shapeType = SHAPE_POLY;
				else
					shapeType = SHAPE_SPHERE;

				int n;
				ss >> n;

				coords.resize(n);
				for (int i = 0; i < n; i++) {
					ss >> coords[i].x >> coords[i].y >> coords[i].z;
				}				


				Transform t;
				t.scale = scale;

				templateParticle = std::move(hullCoordsToMesh(coords).transformed(t));				

			}

			//Instances
			if (defName.length() > 0 && defName.compare(0, defName.length(), line, 0, defName.length()) == CMP_MATCH) {
				std::stringstream ss;
				ss << (line + defName.length() + 1);

				if (shapeType == SHAPE_POLY) {

					int tmp;
					ss >> tmp;

					Transform t;			
					vec3 pos;
					ss >> pos.x >> pos.y >> pos.z;
					ss >> t.rotation[0] >> t.rotation[1] >> t.rotation[2] >> t.rotation[3];

					
					t.translation = pos * scale + vec3(0.5f);

					
					auto instance = std::make_shared<GeometryObject>(templateParticle);
					instance->setTransform(t);					

					if (trim.testIntersection(instance->bounds())) {

						vec3 trimScale = vec3(1.0 / trimRange.x, 1.0 / trimRange.y, 1.0 / trimRange.z);
						t.translation = pos * scale * trimScale + trimScale * vec3(0.5f);
						t.scale = trimScale;

						instance->setTransform(t);

						res.push_back(instance);
					}

					
				}				

			}
		}

		return res;

	}

	FAST_EXPORT size_t getPosFileCount(std::ifstream & stream)
	{
		size_t count = 0;

		size_t pos = stream.tellg();

		const std::string boxS = "boxMatrix";
		char line[4096];
		while (stream.good()) {			
			stream.getline(line, 4096);
			if (boxS.compare(0, boxS.length(), line, 0, boxS.length()) == CMP_MATCH) {
				count++;
			}
		}

		//Reset stream to where it was
		stream.clear();
		stream.seekg(pos);
		

		return count;

	}

}

