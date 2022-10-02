#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "timing.h"

using namespace cl;

char SOURCE_FILE[] = "kernels.cl";
char KERNEL_NAME[] = "task1";

int N = 1 << 24;
int G = N / (1 << 4);
int L = 512;


int main() {
	std::vector<int> sequence(N, 1);
	int nPrimes = 0;

	for (int i = 0; i < N; i++) {
		sequence[i] = i;
	}

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

		Buffer sequenceBuffer = Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, N * sizeof(int), &sequence[0]);
		Buffer nPrimesBuffer = Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int));

		kernel.setArg(0, sequenceBuffer);
		kernel.setArg(1, nPrimesBuffer);
		kernel.setArg(2, N);

		NDRange global(G, 1);
		NDRange local(L, 1);

		queue.enqueueNDRangeKernel(kernel, NullRange, global, local);
		queue.finish();
		queue.enqueueReadBuffer(nPrimesBuffer, CL_TRUE, 0, sizeof(int), &nPrimes);

		std::cout << "Trajanje: " << clock.stop() << " s" << std::endl;
		std::cout << "Broj prostih brojeva: " << nPrimes << std::endl;

	} catch (Error error) {
		std::cout << error.what() << "(" << error.err() << ")" << std::endl;
		std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(cl::Device::getDefault()) << std::endl;
		std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(cl::Device::getDefault()) << std::endl;
		std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault()) << std::endl;
	}

	return 0;
}
