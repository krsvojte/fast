#pragma once
#include <cmath>
#include <type_traits>


using uint64 = unsigned long long;
using uint = unsigned int;
using uchar = unsigned char;


enum PrimitiveType {
	TYPE_FLOAT = 0,	
	TYPE_CHAR,
	TYPE_UCHAR,
	TYPE_INT,
	TYPE_UINT,
	TYPE_UINT64,
	TYPE_FLOAT3,
	TYPE_FLOAT4,	
	TYPE_DOUBLE,
	TYPE_UCHAR3,
	TYPE_UCHAR4,
	TYPE_UNKNOWN
};

template<typename T>
PrimitiveType primitiveTypeof() {
	if (std::is_same<T, float>::value) 
		return TYPE_FLOAT;	
	else if (std::is_same<T, double>::value)
		return TYPE_DOUBLE;
	else if (std::is_same<T, uchar>::value)
		return TYPE_UCHAR;
	else if (std::is_same<T, int>::value)
		return TYPE_INT;
	else if (std::is_same<T, uint>::value)
		return TYPE_UINT;
	else if (std::is_same<T, uint64>::value)
		return TYPE_UINT64;
	/*else if (std::is_same<T, uchar3>::value)
		return TYPE_UCHAR3;
	else if (std::is_same<T, uchar4>::value)
		return TYPE_UCHAR4;*/
	/*else if (std::is_same<T, float3>::value)
		return TYPE_FLOAT3;
	else if (std::is_same<T, float4>::value)
		return TYPE_FLOAT4;*/
	
	return TYPE_UNKNOWN;
}

inline uint primitiveSizeof(PrimitiveType type) {
	switch (type) {
	case TYPE_FLOAT: 
		return sizeof(float);
	case TYPE_DOUBLE:
		return sizeof(double);
	case TYPE_CHAR:
		return sizeof(char);
	case TYPE_UCHAR:
		return sizeof(uchar);
	case TYPE_INT:
		return sizeof(int);
	case TYPE_UINT:
		return sizeof(uint);
	case TYPE_UINT64:
		return sizeof(uint64);
	case TYPE_FLOAT3:
		return sizeof(float) * 3;
	case TYPE_FLOAT4:
		return sizeof(float) * 4;
	case TYPE_UCHAR3:
		return sizeof(uchar) * 3;
	case TYPE_UCHAR4:
		return sizeof(uchar) * 3;
	}
	return 0;
}

inline float primitiveToFloat(PrimitiveType type, void * ptr) {
	switch (type) {
	case TYPE_FLOAT:
		return *((float*)ptr);
	case TYPE_DOUBLE:
		return float(*((double*)ptr));
	case TYPE_CHAR:
		return float(*((char*)ptr));
	case TYPE_UCHAR:
		return float(*((uchar*)ptr));;
	case TYPE_INT:
		return float(*((int*)ptr));;		
	case TYPE_UINT:
		return float(*((uint*)ptr));;
	case TYPE_UINT64:
		return float(*((uint64*)ptr));;	
	}
	return std::nanf("");
}



inline float primitiveToNormFloat(PrimitiveType type, void * ptr) {
	float val = primitiveToFloat(type, ptr);	
	switch (type) {	
	case TYPE_CHAR:
		return val / 127.0f;
	case TYPE_UCHAR:
		return val / 255.0f;
	case TYPE_INT:
		return val / 2147483648.0f;
	case TYPE_UINT:
		return val / float((uint(-1)));
	case TYPE_UINT64:
		return val / float((uint64(-1)));	
	}
	return val;
}

inline double primitiveToDouble(PrimitiveType type, void * ptr) {
	switch (type) {
	case TYPE_FLOAT:
		return double(*((float*)ptr));
	case TYPE_DOUBLE:
		return (*((double*)ptr));
	case TYPE_CHAR:
		return double(*((char*)ptr));
	case TYPE_UCHAR:
		return double(*((uchar*)ptr));
	case TYPE_INT:
		return double(*((int*)ptr));
	case TYPE_UINT:
		return double(*((uint*)ptr));
	case TYPE_UINT64:
		return double(*((uint64*)ptr));
	}

	return std::nan("");
}



enum Dir {
	X_POS = 0,
	X_NEG = 1,
	Y_POS = 2,
	Y_NEG = 3,
	Z_POS = 4,
	Z_NEG = 5,
	DIR_NONE = 6
};


inline uint getDirIndex(Dir dir) {
	switch (dir) {
		case X_POS:
		case X_NEG:
			return 0;
		case Y_POS:
		case Y_NEG:
			return 1;
		case Z_POS:
		case Z_NEG:
			return 2;
		default:
			return uint(-1);
	}
}

inline int getDirSgn(Dir dir) {
	return -((dir % 2) * 2 - 1);
}

inline Dir getDir(int index, int sgn) {
	sgn = (sgn + 1) / 2; // 0 neg, 1 pos
	sgn = 1 - sgn; // 1 neg, 0 pos
	return Dir(index * 2 + sgn);
}

inline Dir getOppositeDir(Dir dir) {
	uint index = getDirIndex(dir);
	return getDir(index, getDirSgn(dir) * -1);
	
}


