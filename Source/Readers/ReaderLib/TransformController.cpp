#include "TransformController.h"

#include <boost\crc.hpp>

#define STREAMID 0

namespace Microsoft { namespace MSR { namespace CNTK {
	unsigned int TransformController::InfoCheck(Sequences & sequence)
	{
		boost::crc_32_type checker;
		for (size_t i = 0; i < sequence.m_data[STREAMID].size(); i++) {
			auto inputSequence = static_cast<DenseSequenceData&>(*sequence.m_data[STREAMID][i]);

			ImageDimensions dimensions(*inputSequence.m_sampleLayout, HWC);
			int columns = static_cast<int>(dimensions.m_width);
			int rows = static_cast<int>(dimensions.m_height);
			int channels = static_cast<int>(dimensions.m_numChannels);

			checker.process_block((char*)inputSequence.m_data, (char*)inputSequence.m_data + columns * rows * channels * sizeof(float));
		}

		unsigned int checksum = checker.checksum();

		return checksum;
	}
}
}
}