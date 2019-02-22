#pragma once



#if defined(_MSC_VER) 
    //  Microsoft 
    #define EXPORT __declspec(dllexport)
    #define IMPORT __declspec(dllimport)
#elif defined(__GNUC__) || defined(__clang__)
    //  GCC
    #define EXPORT __attribute__((visibility("default")))
    #define IMPORT
#else
    //  do nothing and hope for the best?
    #define EXPORT
    #define IMPORT
    #pragma warning Unknown dynamic link import/export semantics.
#endif

#if defined(FASTLIB_EXPORT) 
#   define FAST_EXPORT   EXPORT
#else 
#   define FAST_EXPORT   IMPORT
#endif 