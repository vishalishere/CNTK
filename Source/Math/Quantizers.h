//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once
#include "Basics.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// RawType - input type to the quantizer. Currently CNTK supports float or double as RawType.
// QuantizedType - output type of the quantizer
template <class RawType, class QuantizedType>
class QuantizerBase 
{
public:
    QuantizerBase()
    {
        rangeMax = std::numeric_limits<QuantizedType>::max();
    }
    virtual void Quantize(const Array<RawType>& input, Array<QuantizedType>& output) = 0;
    virtual void Dequantize(const Array<QuantizedType>& input, Array<RawType>& output) = 0;

protected:
    QuantizedType rangeMax;
};

// Symmetric quantizer. 
// Quantization is achieved by 
//    1. Finding the absolute max of values to be quantized.
//    2. Adjusting the absolute max with extraBits parameter.
//    3. Scaling all values in the collection to be within the symmetric range of the QuantizedType
template <class RawType, class QuantizedType>
class SymmetricQuantizer : public QuantizerBase<RawType, QuantizedType>
{
    RawType m_quantizeFactor;
    RawType m_inverseQuantizer;
    RawType m_absMax;
public:
    // elements - collection to be quantized
    // extraBits decreases the quantization normalizer to prevent integer overflow during BLAS routines.
    //     Higher extraBits will decrease precision of quantization, but will make BLAS routines less prone to overflow.
    //     For quantization with shorts, recommended value of extraBits is 1-3.
    // This constructor accepts the collection of RawType to initialize internal quantizer
    // and then apply this quantizer to collections with similar range as the one it was initialized with.
    SymmetricQuantizer(const Array<RawType>& input, size_t extraBits)
    {
        m_absMax = FindAbsMax(input);
        Initialize(m_absMax, extraBits);
    }

    // absoluteMax - the range of the quantizer (normally represents maximum absolute value of the values in the collection to be quantized).
    // extraBits - see comment in another ctor
    SymmetricQuantizer(RawType absoluteMax, size_t extraBits)
    {
        Initialize(absoluteMax, extraBits);
    }

    // Perform quantization of the input collection, put result into pre-allocated output collection
    virtual void Quantize(const Array<RawType>& input, Array<QuantizedType>& output)
    {
        for (size_t i = 0; i < input.size; i++)
        {
#ifdef _DEBUG
            assert(abs(input.elements[i]) <= m_absMax);
#endif
            output.elements[i] = (QuantizedType) round((input.elements[i] * m_quantizeFactor));
        }
    }

    // Accept quantized collection as input, put de-quantization result into pre-allocated output collection.
    virtual void Dequantize(const Array<QuantizedType>& input, Array<RawType>& output)
    {
        for (size_t i = 0; i < input.size; i++)
        {
            output.elements[i] = (RawType)(input.elements[i] * m_inverseQuantizer);
        }
    }

private: 
    // Find absolute maximum value
    RawType FindAbsMax(const Array<RawType>& array)
    {
        RawType maxElem = *std::max_element(array.elements, array.elements + array.size);
        RawType minElem = *std::min_element(array.elements, array.elements + array.size);

        return std::max(maxElem, std::abs(minElem));
    }

    void Initialize(RawType absoluteMax, size_t extraBits)
    {
        RawType shiftedMax = absoluteMax * (1 << extraBits);
        if (shiftedMax == 0)
        {
            LogicError("The absolute max element in the sequence to be quantized is 0.");
        }
        m_absMax = absoluteMax;
        m_quantizeFactor = rangeMax / shiftedMax;
        m_inverseQuantizer = 1 / m_quantizeFactor;
    }
};

}}}