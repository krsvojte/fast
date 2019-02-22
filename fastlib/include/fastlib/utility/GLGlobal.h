#pragma once

#include <fastlib/FastLibDef.h>
#include <functional>
#include <GL/glew.h>

#ifdef _DEBUG

#ifdef _WIN32
#define THIS_FUNCTION __FUNCTION__
#else 
#define THIS_FUNCTION __PRETTY_FUNCTION__
#endif

#define S1(x) #x
#define S2(x) S1(x)
#define THIS_LINE __FILE__ " : " S2(__LINE__)

#define GL(x) x; GLError(THIS_LINE)

FAST_EXPORT void logCerr(const char * label, const char * errtype);

FAST_EXPORT bool GLError(
    const char *label = "",
    const std::function<void(const char *label, const char *errtype)>
        &callback = &logCerr);
#else
#define GL(x) x;
#define GLError(x) false
#endif


FAST_EXPORT bool resetGL();

FAST_EXPORT bool initGLEW();
