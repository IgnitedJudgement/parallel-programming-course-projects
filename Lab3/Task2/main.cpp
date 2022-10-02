#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <numeric>
#include "timing.h"

using namespace cl;

char SOURCE_FILE[] = "kernels.cl";
char KERNEL_NAME[] = "task2";

int N = 1 << 26;
int G = N;
int L = 512;
int len = G / L;

int main() {
	std::vector<double> sequence(N, 0);

	Clock clock;
	std::cout << "start..." << std::endl;
	clock.start();

	Program program;

	try {
		std::ifstream sourceFile(SOURCE_FILE);
		std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
		Program::Sources source(1, std::make_pair(sourceCode.c_str(), sourceCode.length() + 1));

		std::vector<Platform> platforms;
		Platform::get(&platforms);

		cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(platforms[0])(), 0 };
		Context context(CL_DEVICE_TYPE_GPU, cps);

		std::vector<Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

		CommandQueue queue = CommandQueue(context, devices[0]);

		program = Program(context, source);

		program.build(devices);

		Kernel kernel(program, KERNEL_NAME);

		Buffer sequenceBuffer = Buffer(context, CL_MEM_WRITE_ONLY, N * sizeof(double));

		kernel.setArg(0, sequenceBuffer);
		kernel.setArg(1, N);

		NDRange global(G, 1);
		NDRange local(L, 1);

		queue.enqueueNDRangeKernel(kernel, NullRange, global, local);
		queue.finish();
		queue.enqueueReadBuffer(sequenceBuffer, CL_TRUE, 0, N * sizeof(double), &sequence[0]);

		std::cout << "Trajanje: " << clock.stop() << "s" << std::endl;
		std::cout << "Pi: " << std::accumulate(sequence.begin(), sequence.end(), 0.0) << std::endl;

	} catch (Error error) {
		std::cout << error.what() << "(" << error.err() << ")" << std::endl;
		std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(cl::Device::getDefault()) << std::endl;
		std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(cl::Device::getDefault()) << std::endl;
		std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault()) << std::endl;
	}

	return 0;
}
