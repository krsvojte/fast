#pragma once
#include <GL/glew.h>
#include <fastlib/utility/GLGlobal.h>
#include "utility/mathtypes.h"

#include <vector>
#include <iterator>

struct VertexAttrib {
	GLuint index;
	GLint size;
	GLenum type;
	GLboolean normalize;
	GLsizei stride;
	GLvoid * pointer;
};



struct VertexAttribArray {
	VertexAttribArray();
	std::vector<VertexAttrib> attribs;
	std::vector<size_t> byteSizes;
	size_t totalByteSize;
};

bool addAttrib(VertexAttribArray &vaa, GLenum type, size_t size,
	bool normalize = false);



struct VertexData {
	VertexData() {};
	VertexData(vec3 _pos, vec3 _normal, vec2 _uv, vec4 _color);
	GLfloat pos[3];
	GLfloat normal[3];
	GLfloat uv[2];
	GLfloat color[4];

	static const VertexAttribArray vertexAttribArray;	
};

struct VertexDataExtended {
	GLfloat pos[3];
	GLfloat normal[3];
	GLfloat uv[2];
	GLfloat color[4];
	GLfloat tangent[3];
	GLfloat bitangent[3];

	static const VertexAttribArray vertexAttribArray;
};



template <typename T>
class VertexBuffer {
public:

	VertexBuffer(const VertexBuffer<T> &) = delete;
	VertexBuffer & operator = (const VertexBuffer<T> &) = delete;

	
	

	VertexBuffer(GLenum usage = GL_STATIC_DRAW, GLenum primitiveType = GL_TRIANGLES) ;


	//Constructor for buffer allocated outside of this class
	VertexBuffer(
		GLuint vboIndex,
		size_t N,
		GLenum usage = GL_DYNAMIC_DRAW,
		GLenum primitiveType = GL_TRIANGLES		
		);


	~VertexBuffer();
	VertexBuffer(VertexBuffer<T> &&other);
	VertexBuffer & operator = (VertexBuffer<T> &&other);

	size_t size() const;
	
	bool setData(typename std::vector<T>::iterator begin, 
				typename std::vector<T>::iterator end);

	bool clear();

	bool setIndices(void * arr, 
					size_t N, 
					GLenum indexType);

	void setUsage(GLenum usage);

	void setPrimitiveType(GLenum type);

	GLenum primitiveType() const { return m_primitiveType; }
	

	T * map();
	T * mapReadOnly() const;
	void unmap();
	void unmap() const;

	virtual bool render(GLenum mode = 0xAAAAAA, size_t offset = 0, 
				size_t count = std::numeric_limits<std::size_t>::max()) const;

	GLuint getVBOIndex() const {
		return m_buffer;
	}
	

protected:
	void _free();
	bool setAttribs(const VertexAttribArray & vao);

	GLuint m_buffer;
	GLuint m_vao;
	
	GLenum m_usage;
	GLenum m_primitiveType;
	size_t m_size;


	GLuint m_indexBuffer;
	GLenum m_indexType;
	GLbyte m_indexTypeSize;
	size_t m_indexCount;

	bool m_allocated;
};

////////////////////////////////

/*template<typename T>
class IndexedBuffer : public VertexBuffer<T> {
public:

	IndexedBuffer(GLenum usage = GL_STATIC_DRAW);

	template <typename S>
	bool setIndices(typename std::vector<S>::iterator begin,
		typename std::vector<S>::iterator end,
		GLenum indexType) {
		if (begin == end)
			return false;

		m_indexCount = std::distance(begin, end);

		m_indexTypeSize = sizeof(GLbyte);
		if (indexType == GL_UNSIGNED_INT) {
			m_indexTypeSize = sizeof(GLint);
		}
		else if (indexType == GL_UNSIGNED_SHORT) {
			m_indexTypeSize = sizeof(GLshort);
		}

		m_indexType = indexType;
		GLsizei byteSize = static_cast<GLsizei>(m_indexCount) * m_indexTypeSize;

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, byteSize, &begin[0], m_usage);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

		return !GLError(THIS_FUNCTION);

	}

	virtual bool render(GLenum mode = 0xAAAAAA, size_t offset = 0,
		size_t count = std::numeric_limits<std::size_t>::max()) const;
private:
	GLuint m_indexBuffer;
	GLenum m_indexType;
	GLbyte m_indexTypeSize;
	size_t m_indexCount;
};
*/




