#pragma once

#include <fastlib/FastLibDef.h>

#include <fastlib/geometry/Transform.h>
#include <fastlib/geometry/AABB.h>
#include <fastlib/geometry/Geometry.h>

#include <memory>


namespace fast {

	//struct Geometry;

	/*
		Geometric object with transform
		Keeps cached transformed geometry & bounds
	*/
	struct GeometryObject {

		FAST_EXPORT GeometryObject(std::shared_ptr<Geometry> templateGeometry);

		FAST_EXPORT void setTransform(Transform & transform);

		FAST_EXPORT Transform getTransform() const;

		FAST_EXPORT AABB bounds() const;

		FAST_EXPORT const std::unique_ptr<Geometry> & getGeometry() const;

		FAST_EXPORT const std::shared_ptr<Geometry> & getTemplateGeometry() const;


	private:
		std::shared_ptr<Geometry> _templateGeometry;
		Transform _transform;

		mutable bool _boundsDirty = true;
		mutable AABB _bounds;

		mutable bool _geomDirty = true;
		mutable std::unique_ptr<Geometry> _geometryCached;
	};

}