#include "VertexBuffer.h"

#include <cstring>

//template class IndexedBuffer<VertexData>;
//template class IndexedBuffer<VertexDataExtended>;
#ifdef TRACK_GPU_ALLOC
#include <iostream>
#endif

VertexAttribArray::VertexAttribArray()
{
	attribs.reserve(8);
	byteSizes.reserve(8);
	totalByteSize = 0;
}


bool addAttrib(VertexAttribArray &vaa, GLenum type, size_t size, bool normalize /*= false*/)
{
	VertexAttrib a = {
		static_cast<GLuint>(vaa.attribs.size()),
		static_cast<GLint>(size),
		type,
		static_cast<GLboolean>(normalize),
		0,
		nullptr
	};

	size_t byteSize = size;
	switch (type) {
	case GL_FLOAT:
		byteSize *= sizeof(GLfloat);
		break;
	case GL_SHORT:
	case GL_UNSIGNED_SHORT:
		byteSize *= sizeof(GLshort);
		break;
	case GL_INT:
	case GL_UNSIGNED_INT:
		byteSize *= sizeof(GLint);
	case GL_HALF_FLOAT:
		byteSize *= sizeof(GLhalf);
		break;
	default:
		return false;
	}

	vaa.attribs.push_back(a);
	vaa.byteSizes.push_back(byteSize);
	vaa.totalByteSize += byteSize;

	size_t ptr = 0;
	for (size_t i = 0; i < vaa.attribs.size(); i++) {
		VertexAttrib &a = vaa.attribs[i];
		a.stride = static_cast<GLsizei>(vaa.totalByteSize);
		a.pointer = (GLvoid*)ptr;
		ptr += vaa.byteSizes[i];
	}

	return true;
}



template <typename T>
VertexBuffer<T>::VertexBuffer(GLenum usage, GLenum primitiveType) :
	m_primitiveType(primitiveType),
	m_usage(usage), 
	m_buffer(0), 
	m_vao(0), 
	m_indexBuffer(0),
	m_indexCount(0)
{
	
	glGenBuffers(1, &m_buffer);
	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_indexBuffer);

#ifdef TRACK_GPU_ALLOC
	std::cout << "GenBuffer " << m_buffer << ", " << __FILE__ << ":" << __LINE__ << std::endl;
#endif

	//LOG(toString("% Created buffer %", this, m_buffer));

	setAttribs(T::vertexAttribArray);
	m_size = 0;
	GLError(THIS_LINE);
	
	m_allocated = true;

}


template <typename T>
VertexBuffer<T>::VertexBuffer(GLuint vboIndex, size_t N, GLenum usage /*= GL_DYNAMIC_DRAW */, GLenum primitiveType /*= GL_TRIANGLES*/) :
	m_primitiveType(primitiveType),
	m_usage(usage),
	m_buffer(vboIndex),
	m_size(N),
	m_vao(0),
	m_indexBuffer(0),
	m_indexCount(0)
{

	glGenVertexArrays(1, &m_vao);
	glGenBuffers(1, &m_indexBuffer);
	setAttribs(T::vertexAttribArray);

	GLError(THIS_LINE);

	m_allocated = true;

}


template <typename T>
VertexBuffer<T>::~VertexBuffer()
{	
	_free();
}


template <typename T>
void VertexBuffer<T>::_free()
{
	if (m_allocated) {
		glDeleteVertexArrays(1, &m_vao);
		glDeleteBuffers(1, &m_buffer);
#ifdef TRACK_GPU_ALLOC
		std::cout << "DeleteBuffer " << m_buffer << ", " << __FILE__ << ":" << __LINE__ << std::endl;
#endif
		glDeleteBuffers(1, &m_indexBuffer);
		GLError(THIS_LINE);
	}
	m_buffer = 0;
	m_vao = 0;
	m_indexBuffer = 0;
	m_allocated = false;
}


template <typename T>
VertexBuffer<T>::VertexBuffer(VertexBuffer<T> &&other) {	
	memcpy(this, &other, sizeof(other));
	memset(&other, 0, sizeof(VertexBuffer<T>));	
	

/*	m_buffer = other.m_buffer;	
	m_vao = other.m_vao;
	m_size = other.m_size;
	m_usage = other.m_usage;
	m_primitiveType = other.m_primitiveType;

	m_indexBuffer = other.m_indexBuffer;
	m_indexCount = other.m_indexCount;
	m_indexType = other.m_indexType;
	m_indexTypeSize = other.m_indexTypeSize;

	other.m_buffer = 0;
	other.m_vao = 0;
	other.m_indexBuffer = 0;*/
}

template <typename T>
VertexBuffer<T> & VertexBuffer<T>::operator = (VertexBuffer<T> &&other) {
	if (this != &other) {
		this->_free();
		memcpy(this, &other, sizeof(other));
		memset(&other, 0, sizeof(VertexBuffer<T>));
		

		/*m_buffer = other.m_buffer;
		m_vao = other.m_vao;
		m_size = other.m_size;
		m_usage = other.m_usage;
		m_primitiveType = other.m_primitiveType;

		m_indexBuffer = other.m_indexBuffer;
		m_indexCount = other.m_indexCount;
		m_indexType = other.m_indexType;
		m_indexTypeSize = other.m_indexTypeSize;

		other.m_buffer = 0;
		other.m_vao = 0;
		other.m_indexBuffer = 0;*/
	}
	return *this;
}



template <typename T>
void VertexBuffer<T>::setPrimitiveType(GLenum type)
{
	m_primitiveType = type;
}


template <typename T>
size_t VertexBuffer<T>::size() const {
	return m_size;
}

template <typename T>
void VertexBuffer<T>::setUsage(GLenum usage)
{
	m_usage = usage;
}


template <typename T>
bool VertexBuffer<T>::setData( typename std::vector<T>::iterator begin, 
	typename std::vector<T>::iterator end)
{
	if (begin == end)
		return false;

	m_size = std::distance(begin, end);

	glBindBuffer(GL_ARRAY_BUFFER, m_buffer);
	glBufferData(GL_ARRAY_BUFFER, m_size * sizeof(T), &begin[0], m_usage);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return !GLError(THIS_FUNCTION);
}


template <typename T>
bool VertexBuffer<T>::clear()
{
	this->m_size = 0;
	this->m_indexCount = 0;
	/*glBindBuffer(GL_ARRAY_BUFFER, m_buffer);
	glBufferData(GL_ARRAY_BUFFER, 0, nullptr, m_usage);
	glBindBuffer(GL_ARRAY_BUFFER, 0);*/
	return !GLError(THIS_FUNCTION);
}


template <typename T>
bool VertexBuffer<T>::setIndices(void * arr,
	size_t N,
	GLenum indexType) {

	assert(arr != nullptr);

	if (N == 0) return false;

	m_indexCount = N;

	m_indexTypeSize = sizeof(GLbyte);
	if (indexType == GL_UNSIGNED_INT || indexType == GL_INT) {
		indexType = GL_UNSIGNED_INT;
		m_indexTypeSize = sizeof(GLuint);		
	}
	else if (indexType == GL_UNSIGNED_SHORT || indexType == GL_SHORT) {
		indexType = GL_UNSIGNED_SHORT;
		m_indexTypeSize = sizeof(GLshort);
	}

	m_indexType = indexType;
	GLsizei byteSize = static_cast<GLsizei>(m_indexCount) * m_indexTypeSize;

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, byteSize, arr, m_usage);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	return !GLError(THIS_FUNCTION);
}


template <typename T>
bool VertexBuffer<T>::setAttribs(const VertexAttribArray & vao)
{
	assert(vao.totalByteSize > 0);

	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_buffer);	
	GLError(THIS_LINE);

	for(auto & att : vao.attribs){
		glVertexAttribPointer(att.index, att.size, att.type, att.normalize, att.stride, att.pointer);
		glEnableVertexAttribArray(att.index);
	}
	GLError(THIS_LINE);

	glBindVertexArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	return !GLError(THIS_FUNCTION);
}

template <typename T>
void VertexBuffer<T>::unmap()
{
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	GLError(THIS_LINE);
}

template <typename T>
void VertexBuffer<T>::unmap() const
{
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	GLError(THIS_LINE);
}

template <typename T>
T * VertexBuffer<T>::mapReadOnly() const
{
	glBindBuffer(GL_ARRAY_BUFFER, m_buffer);
	GLError(THIS_LINE);
	return reinterpret_cast<T*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_ONLY));
}

template <typename T>
T * VertexBuffer<T>::map()
{
	glBindBuffer(GL_ARRAY_BUFFER, m_buffer);
	GLError(THIS_LINE);
	return reinterpret_cast<T*>(glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE));
}

template <typename T>
bool VertexBuffer<T>::render(
    GLenum mode, size_t offset /*= 0*/,
    size_t count /*= std::numeric_limits<std::size_t>::max()*/) const{
	
	if (!m_size) return false;

	glBindVertexArray(m_vao); GLError(THIS_LINE);	
	glBindBuffer(GL_ARRAY_BUFFER, m_buffer); GLError(THIS_LINE);
	if(m_indexCount > 0)
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer); GLError(THIS_LINE);

	GLError(THIS_LINE);

	//Draw all elements
	if (mode == 0xAAAAAA) {
		mode = m_primitiveType;
	}


	//glPushAttrib(GL_ENABLE_BIT);
	
	if (m_indexCount > 0) {
		if (count == std::numeric_limits<std::size_t>::max())
			count = m_indexCount;

		glDrawElements(mode, static_cast<GLsizei>(count), m_indexType, (const void*)(offset * m_indexTypeSize));
	}
	else {
		if (count == std::numeric_limits<std::size_t>::max())
			count = m_size;

		glDrawArrays(mode, static_cast<GLint>(offset), static_cast<GLsizei>(count));
	}
	//glPopAttrib();

	GLError(THIS_LINE);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);	
	glBindVertexArray(0);	
	
	return !GLError(THIS_FUNCTION);
}




/*
	Indexed buffer
*/

/*
template<typename T>
IndexedBuffer<T>::IndexedBuffer(GLenum usage )
	: VertexBuffer<T>(usage)
{	
	glGenBuffers(1, &m_indexBuffer);
	setAttribs(T::vertexAttribArray);
}*/

/*
template <typename T>
bool IndexedBuffer<T>::setIndices(typename std::vector<T>::iterator begin,
                                  typename std::vector<T>::iterator end,
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
}*/
/*
template <typename T>
bool IndexedBuffer<T>::render(
    GLenum mode  size_t offset ,
    size_t count ) const{
	
	glBindVertexArray(m_vao);
	glBindBuffer(GL_ARRAY_BUFFER, m_buffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_indexBuffer);

	
	if (mode == 0xAAAAAA) {
		mode = m_primitiveType;
	}

	//Draw all elements
	if (count == std::numeric_limits<std::size_t>::max())
		count = m_size;

	glDrawElements(mode, static_cast<GLsizei>(count), m_indexType, (const void*)(offset * m_indexTypeSize));

	

	glBindVertexArray(0);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	return !GLError(THIS_FUNCTION);
}
*/
VertexData::VertexData(vec3 _pos, vec3 _normal, vec2 _uv, vec4 _color)
{
	memcpy(pos, &_pos, sizeof(vec3));
	memcpy(normal, &_normal, sizeof(vec3));
	memcpy(uv, &_uv, sizeof(vec2));
	memcpy(color, &_color, sizeof(vec4));
}



const VertexAttribArray VertexData::vertexAttribArray = []{
	
	VertexAttribArray vao;
	addAttrib(vao, GL_FLOAT, 3); //pos
	addAttrib(vao, GL_FLOAT, 3); //normal
	addAttrib(vao, GL_FLOAT, 2); //tex
	addAttrib(vao, GL_FLOAT, 4); //color rgba

	return vao;
}();


const VertexAttribArray VertexDataExtended::vertexAttribArray = [] {

	VertexAttribArray vao;
	addAttrib(vao, GL_FLOAT, 3); //pos
	addAttrib(vao, GL_FLOAT, 3); //normal
	addAttrib(vao, GL_FLOAT, 2); //tex
	addAttrib(vao, GL_FLOAT, 4); //color rgba
	addAttrib(vao, GL_FLOAT, 3); //tangent
	addAttrib(vao, GL_FLOAT, 3); //bitangent
	return vao;
}();



/*
	Explicit instantiation
*/
template class VertexBuffer<VertexData>;
template class VertexBuffer<VertexDataExtended>;