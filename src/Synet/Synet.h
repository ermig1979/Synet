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
        SynetTensorFormatNhwc, /*!< NHWC format. */
    } SynetTensorFormat;

    /*! @ingroup c_api
        Describes tensor data type.
    */
    typedef enum
    {
        SynetTensorTypeUnknown = -1, /*!< Unknown tensor data type. */
        SynetTensorType32f = 0, /*!< FP32 tensor data type. */
        SynetTensorType32i, /*!< Signed INT32 tensor data type. */
        SynetTensorType8i, /*!< Signed INT8 tensor data type. */
        SynetTensorType8u, /*!< Unsigned INT8 tensor data type. */
        SynetTensorType64i, /*!< Signed INT64 tensor data type. */
        SynetTensorType64u, /*!< Unsigned INT64 tensor data type. */
        SynetTensorTypeBool, /*!< Boolean (1-byte) tensor data type. */
    } SynetTensorType;

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

        \fn SynetBool SynetNetworkEmpty(void * network);

        \short Checks is network model loaded.

        \param [in] network - a network context. It is creted by function ::SynetNetworkInit and released by function ::SynetRelease.
        \return result of checking.
    */
    SYNET_API SynetBool SynetNetworkEmpty(void* network);

    /*! @ingroup c_api

        \fn SynetBool SynetNetworkReshape(void* network, size_t width, size_t height, size_t batch);

        \short Reshapes previously loaded network model.

        \param [in] network - a network context. It is creted by function ::SynetNetworkInit and released by function ::SynetRelease.
        \param [in] width - a width of input tensor.
        \param [in] height - a height of input tensor.
        \param [in] batch - a batch size of input tensor.
        \return result of the operation.
    */
    SYNET_API SynetBool SynetNetworkReshape(void* network, size_t width, size_t height, size_t batch);

    /*! @ingroup c_api

        \fn  SynetBool SynetNetworkSetBatch(void* network, size_t batch);

        \short Sets batch in previously loaded network model.

        \param [in] network - a network context. It is creted by function ::SynetNetworkInit and released by function ::SynetRelease.
        \param [in] batch - a batch size of input tensor.
        \return result of the operation.
    */
    SYNET_API SynetBool SynetNetworkSetBatch(void* network, size_t batch);

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
        \return a pointer to given input tensor. It is valid until you reload or reshape model.
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
        \return a pointer to given output tensor. It is valid until you reload or reshape model. Inference rewrites output tensors.
    */
    SYNET_API void* SynetNetworkDst(void* network, size_t index);

    /*! @ingroup c_api

        \fn void * SynetNetworkDstByName(void* network, const char * name);

        \short Gets pointer to output tensor with given name.

        \param [in] network - a network context. It is creted by function ::SynetNetworkInit and released by function ::SynetRelease.
        \param [in] name - an output tensor name.
        \return a pointer to given output tensor. It is valid until you reload or reshape model. Inference rewrites output tensors.
    */
    SYNET_API void* SynetNetworkDstByName(void* network, const char * name);

    /*! @ingroup c_api

        \fn void SynetNetworkCompactWeight(void* network);

        \short Reduces memory usage by model network. After calling of this function model can't be reshaped.

        \param [in] network - a network context. It is creted by function ::SynetNetworkInit and released by function ::SynetRelease.
    */
    SYNET_API void SynetNetworkCompactWeight(void* network);

    /*! @ingroup c_api

        \fn void SynetNetworkForward(void* network);

        \short Performs network model inference.

        \param [in] network - a network context. It is creted by function ::SynetNetworkInit and released by function ::SynetRelease.
    */
    SYNET_API void SynetNetworkForward(void* network);

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

        \fn SynetTensorFormat SynetTensorFormatGet(void* tensor);

        \short Gets tensor format.

        \param [in] tensor - a tensor context. It is getted by functions ::SynetNetworkSrc or ::SynetNetworkDst.
        \return a tensor format.
    */
    SYNET_API SynetTensorFormat SynetTensorFormatGet(void* tensor);

    /*! @ingroup c_api

        \fn SynetTensorType SynetTensorTypeGet(void* tensor);

        \short Gets tensor data type.

        \param [in] tensor - a tensor context. It is getted by functions ::SynetNetworkSrc or ::SynetNetworkDst.
        \return a tensor data type.
    */
    SYNET_API SynetTensorType SynetTensorTypeGet(void* tensor);

    /*! @ingroup c_api

        \fn const char * SynetTensorName(void* tensor);

        \short Gets tensor name.

        \param [in] tensor - a tensor context. It is getted by functions ::SynetNetworkSrc or ::SynetNetworkDst.
        \return a tensor name.
    */
    SYNET_API const char * SynetTensorName(void* tensor);

    /*! @ingroup c_api

        \fn uint8_t * SynetTensorData(void* tensor);

        \short Gets pointer to tensor data.

        \param [in] tensor - a tensor context. It is getted by functions ::SynetNetworkSrc or ::SynetNetworkDst.
        \return a pointer to tensor data.
    */
    SYNET_API uint8_t * SynetTensorData(void* tensor);

#ifdef __cplusplus
}
#endif//__cplusplus

#endif //__Synet_h__
