#include "geometry/GeometryObject.h"
#include "geometry/Geometry.h"

namespace fast {

	FAST_EXPORT fast::GeometryObject::GeometryObject(std::shared_ptr<Geometry> geom) : _templateGeometry(geom)
	{

	}


	void GeometryObject::setTransform(Transform & transform)
	{
		_geomDirty = true;
		_boundsDirty = true;
		_transform = transform;
	}

	FAST_EXPORT Transform GeometryObject::getTransform() const
	{
		return _transform;
	}

	FAST_EXPORT AABB GeometryObject::bounds() const
	{
		if (_boundsDirty || _geomDirty) {
			_bounds = getGeometry()->bounds();
			_boundsDirty = false;
		}
		return _bounds;
	}

	FAST_EXPORT const std::unique_ptr<fast::Geometry> & GeometryObject::getGeometry() const
	{
		if (_geomDirty) {
			_geometryCached = _templateGeometry->transformed(_transform);
			_geomDirty = false;
		}
		return _geometryCached;
	}

	FAST_EXPORT const std::shared_ptr<fast::Geometry> & GeometryObject::getTemplateGeometry() const
	{
		return _templateGeometry;
	}

}
