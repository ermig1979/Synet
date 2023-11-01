/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2023 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#ifndef __Synet_h__
#define __Synet_h__

#include <stdint.h>

/*! @ingroup c_api
    Describes boolean type.
*/
typedef enum
{
    SynetFalse = 0, /*!< False value. */
    SynetTrue = 1, /*!< True value. */
} SynetBool;

#if defined(_WIN32) && !defined(SYNET_STATIC)
#  ifdef SYNET_EXPORTS
#    define SYNET_API __declspec(dllexport)
#  else
#    define SYNET_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) && defined(SYNET_HIDE_INTERNAL)
#    define SYNET_API __attribute__ ((visibility ("default")))
#else
#    define SYNET_API
#endif

#ifdef __cplusplus
extern "C"
{
#endif//__cplusplus

    /*! @ingroup c_api

        \fn const char * SynetVersion();

        \short Gets version of %Synet Framework.

        \return string with version of %Synet.
    */
    SYNET_API const char* SynetVersion();

    /*! @ingroup c_api

        \fn void SiynetRelease(void * context);

        \short Releases context created with using of Synet Framework API.

        \param [in] context - a context to be released.
    */ 
    SYNET_API void SynetRelease(void* context);

#ifdef __cplusplus
}
#endif//__cplusplus

#endif //__Synet_h__
