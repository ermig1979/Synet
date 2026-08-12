/*
* Synet Framework (http://github.com/ermig1979/Synet).
*
* Copyright (c) 2018-2024 Yermalayeu Ihar.
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
        Describes a boolean value used by the C API.

        Function parameters of this type enable or disable an option, and functions
        returning this type report success/failure or another binary state.
    */
    typedef enum
    {
        SynetFalse = 0, /*!< False, disabled or failed state. */
        SynetTrue = 1, /*!< True, enabled or successful state. */
    } SynetBool;

    /*! @ingroup c_api
        Describes console logger verbosity used by ::SynetSetConsoleLogLevel.

        Higher values include messages of all lower levels. ::SynetLogNone disables
        console logging. Levels map to the internal Cpl logger writers attached to
        <tt>std::cout</tt>.
    */
    typedef enum
    {
        SynetLogNone = 0, /*!< Disable console logging (silence mode). */
        SynetLogError,  /*!< Log error messages. */
        SynetLogWarning, /*!< Log warnings and more severe messages. */
        SynetLogInfo, /*!< Log informational messages and above. */
        SynetLogVerbose, /*!< Log verbose diagnostic messages and above. */
        SynetLogDebug, /*!< Log full debug information. */
    } SynetLogLevel;

    /*! @ingroup c_api
        Describes tensor memory layout used by network tensors.

        Most network tensors are 4D with dimensions batch (N), channels (C), height (H)
        and width (W). Layout selects whether channels are the outer or inner spatial
        axis. ::SynetTensorFormatUnknown means the layout is unknown or irrelevant.
    */
    typedef enum
    {
        SynetTensorFormatUnknown = -1, /*!< Unknown or layout-independent tensor format. */
        SynetTensorFormatNchw = 0, /*!< NCHW layout: offset = ((n*C + c)*H + h)*W + w. */
        SynetTensorFormatNhwc, /*!< NHWC layout: offset = ((n*H + h)*W + w)*C + c. */
    } SynetTensorFormat;

    /*! @ingroup c_api
        Describes tensor element data type used by network tensors.

        The value defines interpretation and size of each tensor element. Reduced
        precision values ::SynetTensorType16b and ::SynetTensorType16f are stored in
        16-bit containers.
    */
    typedef enum
    {
        SynetTensorTypeUnknown = -1, /*!< Unknown tensor data type. */
        SynetTensorType32f = 0, /*!< 32-bit floating point (single precision). */
        SynetTensorType32i, /*!< 32-bit signed integer. */
        SynetTensorType8i, /*!< 8-bit signed integer. */
        SynetTensorType8u, /*!< 8-bit unsigned integer. */
        SynetTensorType64i, /*!< 64-bit signed integer. */
        SynetTensorType64u, /*!< 64-bit unsigned integer. */
        SynetTensorTypeBool, /*!< Boolean value stored in one byte. */
        SynetTensorType16b, /*!< 16-bit BFloat16 (Brain Floating Point) stored in a 16-bit container. */
        SynetTensorType16f, /*!< 16-bit floating point (Half Precision) stored in a 16-bit container. */
    } SynetTensorType;

    /*! @ingroup c_api

        \fn const char * SynetVersion();

        \short Gets version of %Synet Framework.

        Returns a pointer to a null-terminated string that encodes the framework version.
        The string is produced at build time from the project version file and typically
        has the form <tt>"major.minor.release[.date.branch-sha]"</tt> (for example
        <tt>"1.0.8.2026-08-12.HEAD-d3530fc0"</tt>).

        The returned pointer is valid for the lifetime of the process and must not be freed.

        \return a pointer to a static null-terminated string with the version of %Synet Framework.
    */
    SYNET_API const char* SynetVersion();

    /*! @ingroup c_api

        \fn void SynetSetConsoleLogLevel(SynetLogLevel level);

        \short Sets the console (<tt>std::cout</tt>) logger level of %Synet Framework.

        Replaces any previously installed console log writer with a writer that accepts
        messages at or above \a level. Pass ::SynetLogNone to suppress console output.
        See ::SynetLogLevel for the available levels.

        \param [in] level - a console logger level (see ::SynetLogLevel).
    */
    SYNET_API void SynetSetConsoleLogLevel(SynetLogLevel level);

    /*! @ingroup c_api

        \fn void SynetRelease(void * context);

        \short Destroys an opaque context object created by the Synet Framework C API.

        Releases any context returned by a Synet Framework context-creation function,
        currently ::SynetNetworkInit. Internally the function performs a polymorphic
        \c delete through the virtual destructor of the internal \c Deletable base class.

        Passing \c NULL is safe and has no effect, consistent with the behaviour of a
        C++ \c delete expression on a null pointer.

        \note Passing a pointer that was not returned by a Synet Framework context-creation
              function (for example a tensor pointer from ::SynetNetworkSrc / ::SynetNetworkDst,
              or a pointer from \c malloc / \c new) is undefined behaviour.

        \param [in] context - a pointer to the context to be released, or \c NULL.
    */ 
    SYNET_API void SynetRelease(void* context);

    /*! @ingroup c_api

        \fn void * SynetNetworkInit();

        \short Creates an empty Synet network context.

        Allocates a network object that must later be filled with ::SynetNetworkLoad
        before inference. The returned pointer is opaque for C clients and must be
        released with ::SynetRelease when it is no longer needed.

        Usage example:
        \verbatim
        void* network = SynetNetworkInit();
        if (SynetNetworkLoad(network, "model.xml", "model.bin") == SynetTrue)
        {
            // fill inputs, run SynetNetworkForward, read outputs
        }
        SynetRelease(network);
        \endverbatim

        \return a pointer to a network context. It must be released with ::SynetRelease.
    */
    SYNET_API void * SynetNetworkInit();

    /*! @ingroup c_api

        \fn SynetBool SynetNetworkLoad(void * network, const char * model, const char* weight);

        \short Loads a Synet model and its binary weights into a network context.

        Clears any previously loaded model in \a network, parses the XML model
        description from \a model, creates layers and reads binary weights from
        \a weight. On success the network is ready for reshape/batch changes,
        input filling and ::SynetNetworkForward.

        \param [in, out] network - a network context created by ::SynetNetworkInit and released by ::SynetRelease.
        \param [in] model - a path to the Synet model description file (XML format).
        \param [in] weight - a path to the Synet model binary weights file.
        \return ::SynetTrue on success, ::SynetFalse on failure.
    */
    SYNET_API SynetBool SynetNetworkLoad(void * network, const char * model, const char* weight);

    /*! @ingroup c_api

        \fn SynetBool SynetNetworkEmpty(void * network);

        \short Checks whether a network context has no loaded model.

        Returns ::SynetTrue when \a network has not been successfully loaded (or was
        cleared), and ::SynetFalse after a successful ::SynetNetworkLoad.

        \param [in] network - a network context created by ::SynetNetworkInit and released by ::SynetRelease.
        \return ::SynetTrue if the network is empty, ::SynetFalse if a model is loaded.
    */
    SYNET_API SynetBool SynetNetworkEmpty(void* network);

    /*! @ingroup c_api

        \fn SynetBool SynetNetworkReshape(void* network, size_t width, size_t height, size_t batch);

        \short Reshapes a previously loaded network for a new input size and batch.

        Updates the single 4D input tensor to the given \a width, \a height and
        \a batch according to the model tensor format (::SynetTensorFormatNchw or
        ::SynetTensorFormatNhwc), then rebuilds intermediate tensors. The network must
        already be loaded, must have exactly one input, and that input must be
        resizable (or already marked with dynamic spatial dimensions).

        \param [in] network - a network context created by ::SynetNetworkInit and released by ::SynetRelease.
        \param [in] width - a width of the input tensor.
        \param [in] height - a height of the input tensor.
        \param [in] batch - a batch size of the input tensor.
        \return ::SynetTrue on success, ::SynetFalse on failure.
    */
    SYNET_API SynetBool SynetNetworkReshape(void* network, size_t width, size_t height, size_t batch);

    /*! @ingroup c_api

        \fn SynetBool SynetNetworkSetBatch(void* network, size_t batch);

        \short Sets the batch size of a previously loaded network model.

        Changes only the batch (N) dimension of the single network input and rebuilds
        intermediate tensors. Spatial dimensions are left unchanged. The network must
        already be loaded and must have exactly one input.

        \param [in] network - a network context created by ::SynetNetworkInit and released by ::SynetRelease.
        \param [in] batch - a batch size of the input tensor.
        \return ::SynetTrue on success, ::SynetFalse on failure.
    */
    SYNET_API SynetBool SynetNetworkSetBatch(void* network, size_t batch);

    /*! @ingroup c_api

        \fn size_t SynetNetworkSrcSize(void* network);

        \short Gets the number of network input tensors.

        \param [in] network - a network context created by ::SynetNetworkInit and released by ::SynetRelease.
        \return the number of network input tensors.
    */
    SYNET_API size_t SynetNetworkSrcSize(void* network);

    /*! @ingroup c_api

        \fn void * SynetNetworkSrc(void* network, size_t index);

        \short Gets a pointer to the input tensor at the given index.

        The returned opaque tensor pointer can be queried with ::SynetTensorCount,
        ::SynetTensorAxis, ::SynetTensorFormatGet, ::SynetTensorTypeGet,
        ::SynetTensorName and ::SynetTensorData. It remains valid until the model is
        reloaded or reshaped. Do not release it with ::SynetRelease.

        \param [in] network - a network context created by ::SynetNetworkInit and released by ::SynetRelease.
        \param [in] index - a zero-based index of the input tensor; must be less than ::SynetNetworkSrcSize.
        \return a pointer to the input tensor, or undefined if \a index is out of range.
    */
    SYNET_API void * SynetNetworkSrc(void* network, size_t index);

    /*! @ingroup c_api

        \fn size_t SynetNetworkDstSize(void* network);

        \short Gets the number of network output tensors.

        \param [in] network - a network context created by ::SynetNetworkInit and released by ::SynetRelease.
        \return the number of network output tensors.
    */
    SYNET_API size_t SynetNetworkDstSize(void* network);

    /*! @ingroup c_api

        \fn void * SynetNetworkDst(void* network, size_t index);

        \short Gets a pointer to the output tensor at the given index.

        The returned opaque tensor pointer can be queried with the ::SynetTensor*
        accessors. It remains valid until the model is reloaded or reshaped.
        ::SynetNetworkForward overwrites output tensor contents. Do not release the
        pointer with ::SynetRelease.

        \param [in] network - a network context created by ::SynetNetworkInit and released by ::SynetRelease.
        \param [in] index - a zero-based index of the output tensor; must be less than ::SynetNetworkDstSize.
        \return a pointer to the output tensor, or undefined if \a index is out of range.
    */
    SYNET_API void* SynetNetworkDst(void* network, size_t index);

    /*! @ingroup c_api

        \fn void * SynetNetworkDstByName(void* network, const char * name);

        \short Gets a pointer to the output tensor with the given name.

        Searches network outputs for a tensor whose name equals \a name. The returned
        pointer has the same lifetime rules as ::SynetNetworkDst: it is valid until
        reload/reshape, and inference overwrites its data. Do not release it with
        ::SynetRelease.

        \param [in] network - a network context created by ::SynetNetworkInit and released by ::SynetRelease.
        \param [in] name - a null-terminated output tensor name.
        \return a pointer to the matching output tensor, or \c NULL if no output with that name exists.
    */
    SYNET_API void* SynetNetworkDstByName(void* network, const char * name);

    /*! @ingroup c_api

        \fn void SynetNetworkCompactWeight(void* network);

        \short Reduces memory used by network weights after loading.

        Compacts layer weight storage and may clear unused constant tensors. Call this
        after ::SynetNetworkLoad (and any required reshape/batch setup) when the model
        shape is final. After this call the network should not be reshaped.

        \param [in] network - a network context created by ::SynetNetworkInit and released by ::SynetRelease.
    */
    SYNET_API void SynetNetworkCompactWeight(void* network);

    /*! @ingroup c_api

        \fn void SynetNetworkForward(void* network);

        \short Runs network inference.

        Executes all network stages for the default thread. Input tensors obtained via
        ::SynetNetworkSrc must be filled before calling this function. Output tensors
        obtained via ::SynetNetworkDst / ::SynetNetworkDstByName are updated with the
        inference results.

        \param [in] network - a network context created by ::SynetNetworkInit and released by ::SynetRelease.
    */
    SYNET_API void SynetNetworkForward(void* network);

    /*! @ingroup c_api

        \fn size_t SynetTensorCount(void* tensor);

        \short Gets the number of dimensions (rank) of a tensor.

        \param [in] tensor - a tensor pointer obtained from ::SynetNetworkSrc, ::SynetNetworkDst or ::SynetNetworkDstByName.
        \return the number of dimensions of the tensor.
    */
    SYNET_API size_t SynetTensorCount(void* tensor);

    /*! @ingroup c_api

        \fn size_t SynetTensorAxis(void* tensor, ptrdiff_t axis);

        \short Gets the size of a given tensor dimension.

        \a axis may be negative and is then counted from the end (for example \c -1
        selects the last dimension), matching the C++ tensor \c Axis helper.

        \param [in] tensor - a tensor pointer obtained from ::SynetNetworkSrc, ::SynetNetworkDst or ::SynetNetworkDstByName.
        \param [in] axis - an index of the tensor dimension; may be negative.
        \return the size of the selected dimension.
    */
    SYNET_API size_t SynetTensorAxis(void* tensor, ptrdiff_t axis);

    /*! @ingroup c_api

        \fn SynetTensorFormat SynetTensorFormatGet(void* tensor);

        \short Gets the memory layout of a tensor.

        \param [in] tensor - a tensor pointer obtained from ::SynetNetworkSrc, ::SynetNetworkDst or ::SynetNetworkDstByName.
        \return the tensor format (see ::SynetTensorFormat).
    */
    SYNET_API SynetTensorFormat SynetTensorFormatGet(void* tensor);

    /*! @ingroup c_api

        \fn SynetTensorType SynetTensorTypeGet(void* tensor);

        \short Gets the element data type of a tensor.

        \param [in] tensor - a tensor pointer obtained from ::SynetNetworkSrc, ::SynetNetworkDst or ::SynetNetworkDstByName.
        \return the tensor data type (see ::SynetTensorType).
    */
    SYNET_API SynetTensorType SynetTensorTypeGet(void* tensor);

    /*! @ingroup c_api

        \fn const char * SynetTensorName(void* tensor);

        \short Gets the name of a tensor.

        Returns a pointer to a null-terminated string owned by the tensor/network.
        The pointer remains valid until the owning network is reloaded, reshaped or
        released, and must not be freed by the caller.

        \param [in] tensor - a tensor pointer obtained from ::SynetNetworkSrc, ::SynetNetworkDst or ::SynetNetworkDstByName.
        \return a pointer to the tensor name.
    */
    SYNET_API const char * SynetTensorName(void* tensor);

    /*! @ingroup c_api

        \fn uint8_t * SynetTensorData(void* tensor);

        \short Gets a pointer to the raw tensor data buffer.

        The buffer contains tightly packed elements whose type is reported by
        ::SynetTensorTypeGet. For input tensors, write preprocessed values here before
        ::SynetNetworkForward. For output tensors, read results after inference.
        The pointer remains valid until the owning network is reloaded, reshaped or
        released, and must not be freed by the caller.

        \param [in] tensor - a tensor pointer obtained from ::SynetNetworkSrc, ::SynetNetworkDst or ::SynetNetworkDstByName.
        \return a pointer to the tensor data buffer.
    */
    SYNET_API uint8_t * SynetTensorData(void* tensor);

#ifdef __cplusplus
}
#endif//__cplusplus

#endif //__Synet_h__
