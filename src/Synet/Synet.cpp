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

#include "Synet/Network.h"

#if defined(_WIN32) && !defined(SYNET_STATIC)

#define SYNET_EXPORTS
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

BOOL APIENTRY DllMain(HMODULE hModule, DWORD dwReasonForCall, LPVOID lpReserved)
{
    switch (dwReasonForCall)
    {
    case DLL_PROCESS_DETACH:
    case DLL_PROCESS_ATTACH:
    case DLL_THREAD_ATTACH:
    case DLL_THREAD_DETACH:
        return TRUE;
    }
    return TRUE;
}
#endif//_WIN32

#include "Synet/Synet.h"

SYNET_API const char* SynetVersion()
{
    return SYNET_VERSION;
}

SYNET_API void SynetSetConsoleLogLevel(SynetLogLevel level)
{
    static int id = -1;
    if (id >= 0)
        Cpl::Log::Global().RemoveWriter(id);
    id = Cpl::Log::Global().AddStdWriter((Cpl::Log::Level)level);
    Cpl::Log::Global().SetFlags(Cpl::Log::BashFlags);
}

SYNET_API void SynetRelease(void* context)
{
    delete (Synet::Deletable*)context;
}

//-------------------------------------------------------------------------------------------------

SYNET_API void* SynetNetworkInit()
{
    return new Synet::Network();
}

SYNET_API SynetBool SynetNetworkLoad(void* network, const char* model, const char* weight)
{
    return ((Synet::Network*)network)->Load(model, weight) ? SynetTrue : SynetFalse;
}

SYNET_API SynetBool SynetNetworkEmpty(void* network)
{
    return ((Synet::Network*)network)->Empty() ? SynetTrue : SynetFalse;
}

SYNET_API SynetBool SynetNetworkReshape(void* network, size_t width, size_t height, size_t batch)
{
    return ((Synet::Network*)network)->Reshape(width, height, batch) ? SynetTrue : SynetFalse;
}

SYNET_API SynetBool SynetNetworkSetBatch(void* network, size_t batch)
{
    return ((Synet::Network*)network)->SetBatch(batch) ? SynetTrue : SynetFalse;
}

SYNET_API size_t SynetNetworkSrcSize(void* network)
{
    return ((Synet::Network*)network)->Src().size();
}

SYNET_API void* SynetNetworkSrc(void* network, size_t index)
{
    return ((Synet::Network*)network)->Src()[index];
}

SYNET_API size_t SynetNetworkDstSize(void* network)
{
    return ((Synet::Network*)network)->Dst().size();
}

SYNET_API void* SynetNetworkDst(void* network, size_t index)
{
    return ((Synet::Network*)network)->Dst()[index];
}

SYNET_API void* SynetNetworkDstByName(void* network, const char* name)
{
    return (void*)(((Synet::Network*)network)->Dst(name));
}

SYNET_API void SynetNetworkCompactWeight(void* network)
{
    ((Synet::Network*)network)->CompactWeight();
}

SYNET_API void SynetNetworkForward(void* network)
{
    ((Synet::Network*)network)->Forward();
}

//-------------------------------------------------------------------------------------------------

SYNET_API size_t SynetTensorCount(void* tensor)
{
    return ((Synet::Tensor<float>*)tensor)->Count();
}

SYNET_API size_t SynetTensorAxis(void* tensor, ptrdiff_t axis)
{
    return ((Synet::Tensor<float>*)tensor)->Axis(axis);
}

SYNET_API SynetTensorFormat SynetTensorFormatGet(void* tensor)
{
    return (SynetTensorFormat)((Synet::Tensor<float>*)tensor)->Format();
}

SYNET_API SynetTensorType SynetTensorTypeGet(void* tensor)
{
    return (SynetTensorType)((Synet::Tensor<float>*)tensor)->GetType();
}

SYNET_API const char* SynetTensorName(void* tensor)
{
    return ((Synet::Tensor<float>*)tensor)->Name().c_str();
}

SYNET_API uint8_t* SynetTensorData(void* tensor)
{
    return ((Synet::Tensor<float>*)tensor)->RawData();
}
