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
        Describes boolean type.
    */
    typedef enum
    {
        SynetFalse = 0, /*!< False value. */
        SynetTrue = 1, /*!< True value. */
    } SynetBool;

    /*! @ingroup c_api
        Describes tensor format.
    */
    typedef enum
    {
        SynetTensorFormatUnknown = -1, /*!< Unknown format. */
        SynetTensorFormatNchw = 0, /*!< NCHW format. */
        SynetTensorFormatNhwc= 1, /*!< NHWC format. */
    } SynetTensorFormatType;

    /*! @ingroup c_api

        \fn const char * SynetVersion();

        \short Gets version of %Synet Framework.

        \return string with version of %Synet.
    */
    SYNET_API const char* SynetVersion();

    /*! @ingroup c_api

        \fn void SynetRelease(void * context);

        \short Releases context created with using of Synet Framework API.

        \param [in] context - a context to be released.
    */ 
    SYNET_API void SynetRelease(void* context);

    /*! @ingroup c_api

        \fn void * SynetNetworkInit();

        \short Creates Synet Network.

        \return a pointer to Network. It must be released with using of function ::SynetRelease.
    */
    SYNET_API void * SynetNetworkInit();

    /*! @ingroup c_api

        \fn SynetBool SynetNetworkLoad(void * network, const char * model, const char* weight);

        \short Loads Synet model from files.

        \param [in, out] network - a network context. It is creted by function ::SynetNetworkInit and released by function ::SynetRelease.
        \param [in] model - a path to Synet model description (in XML format).
        \param [in] weight - a path to Synet model binary weights.
        \return result of model loading.
    */
    SYNET_API SynetBool SynetNetworkLoad(void * network, const char * model, const char* weight);

    /*! @ingroup c_api

        \fn size_t SynetNetworkSrcSize(void* network);

        \short Gets number of network inputs.

        \param [in] network - a network context. It is creted by function ::SynetNetworkInit and released by function ::SynetRelease.
        \return a number of network inputs.
    */
    SYNET_API size_t SynetNetworkSrcSize(void* network);

    /*! @ingroup c_api

        \fn void * SynetNetworkSrc(void* network, size_t index);

        \short Gets pointer to given input tensor.

        \param [in] network - a network context. It is creted by function ::SynetNetworkInit and released by function ::SynetRelease.
        \param [in] index - an index of input tensor.
        \return a pointer to given input tensor. It valid until ...
    */
    SYNET_API void * SynetNetworkSrc(void* network, size_t index);

    /*! @ingroup c_api

        \fn size_t SynetNetworkDstSize(void* network);

        \short Gets number of network outputs.

        \param [in,out] network - a network context. It is creted by function ::SynetNetworkInit and released by function ::SynetRelease.
        \return a number of network outputs.
    */
    SYNET_API size_t SynetNetworkDstSize(void* network);

    /*! @ingroup c_api

        \fn void * SynetNetworkDst(void* network, size_t index);

        \short Gets pointer to given output tensor.

        \param [in] network - a network context. It is creted by function ::SynetNetworkInit and released by function ::SynetRelease.
        \param [in] index - an index of output tensor.
        \return a pointer to given output tensor. It valid until ...
    */
    SYNET_API void* SynetNetworkDst(void* network, size_t index);

    /*! @ingroup c_api

        \fn size_t SynetTensorCount(void* tensor);

        \short Gets number of dimensions of tensor.

        \param [in] tensor - a tensor context. It is getted by functions ::SynetNetworkSrc or ::SynetNetworkDst.
        \return a number of dimensions of tensor.
    */
    SYNET_API size_t SynetTensorCount(void* tensor);

    /*! @ingroup c_api

        \fn size_t SynetTensorAxis(void* tensor, ptrdiff_t axis);

        \short Gets size of given dimension of tensor.

        \param [in] tensor - a tensor context. It is getted by functions ::SynetNetworkSrc or ::SynetNetworkDst.
        \param [in] axis - an index of tensor dimension. Can be negative.
        \return a size of given dimension of tensor.
    */
    SYNET_API size_t SynetTensorAxis(void* tensor, ptrdiff_t axis);

    /*! @ingroup c_api

        \fn SynetTensorFormatType SynetTensorFormat(void*  tensor);

        \short Gets size of given dimension of tensor.

        \param [in] tensor - a tensor context. It is getted by functions ::SynetNetworkSrc or ::SynetNetworkDst.
        \return a tensor format.
    */
    SYNET_API SynetTensorFormatType SynetTensorFormat(void* tensor);

    /*! @ingroup c_api

        \fn const char * SynetTensorName(void* tensor);

        \short Gets tensor name.

        \param [in] tensor - a tensor context. It is getted by functions ::SynetNetworkSrc or ::SynetNetworkDst.
        \return a tensor name.
    */
    SYNET_API const char * SynetTensorName(void* tensor);

#ifdef __cplusplus
}
#endif//__cplusplus

#endif //__Synet_h__
