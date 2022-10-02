#define __CL_ENABLE_EXCEPTIONS

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "arraymalloc.h"
#include "boundary.h"
#include "jacobi.h"
#include "cfdio.h"

#include <CL/cl.hpp>
#include <utility>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "timing.h"

using namespace cl;

char SOURCE_FILE[] = "kernels.cl";
char KERNEL_NAME[] = "task3";

int main(int argc, char** argv) {
	int printfreq = 1000;
	double error, bnorm;
	double tolerance = 0;

	double* psi;
	double* psitmp;

	int scalefactor, numiter;

	int bbase = 10;
	int hbase = 15;
	int wbase = 5;
	int mbase = 32;
	int nbase = 32;

	int irrotational = 1, checkerr = 0;

	int m, n, b, h, w;
	int iter;
	int i, j;

	double tstart, tstop, ttot, titer;

	if (tolerance > 0) {
		checkerr = 1;
	}

	if (argc < 3 || argc > 4) {
		printf("Usage: cfd <scale> <numiter>\n");
		return 0;
	}

	scalefactor = atoi(argv[1]);
	numiter = atoi(argv[2]);

	if (!checkerr) {
		printf("Scale Factor = %i, iterations = %i\n", scalefactor, numiter);
	}
	else {
		printf("Scale Factor = %i, iterations = %i, tolerance= %g\n", scalefactor, numiter, tolerance);
	}

	printf("Irrotational flow\n");

	b = bbase * scalefactor;
	h = hbase * scalefactor;
	w = wbase * scalefactor;
	m = mbase * scalefactor;
	n = nbase * scalefactor;

	printf("Running CFD on %d x %d grid in serial\n", m, n);

	psi = (double*) malloc((m + 2) * (n + 2) * sizeof(double));
	psitmp = (double*) malloc((m + 2) * (n + 2) * sizeof(double));

	for (i = 0; i < m + 2; i++) {
		for (j = 0; j < n + 2; j++) {
			psi[i * (m + 2) + j] = 0;
		}
	}

	boundarypsi(psi, m, n, b, h, w);

	bnorm = 0;

	for (i = 0; i < m + 2; i++) {
		for (j = 0; j < n + 2; j++) {
			bnorm += psi[i * (m + 2) + j] * psi[i * (m + 2) + j];
		}
	}

	bnorm = sqrt(bnorm);

	printf("\nStarting main loop...\n\n");
	tstart = gettime();

	// MAIN PROGRAM
	int N = (m + 2) * (n + 2);
	int L = 64;

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

		Buffer inputBuffer = Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(double), &psi[0]);
		Buffer outputBuffer = Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, N * sizeof(double), &psi[0]);

		NDRange global(m, n);
		NDRange local(L, 1);

		for (iter = 1; iter <= numiter; iter++) {
			kernel.setArg(0, inputBuffer);
			kernel.setArg(1, outputBuffer);
			kernel.setArg(2, m);
			kernel.setArg(3, n);

			queue.enqueueNDRangeKernel(kernel, NullRange, global, local);
			queue.finish();

			if (checkerr || iter == numiter) {
				queue.enqueueReadBuffer(inputBuffer, CL_TRUE, 0, N * sizeof(double), &psi[0]);
				queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, N * sizeof(double), &psitmp[0]);

				error = deltasq(psitmp, psi, m, n);
				error = sqrt(error);
				error = error / bnorm;
			}

			if (checkerr) {
				if (error < tolerance) {
					printf("Converged on iteration %d\n", iter);
					break;
				}
			}

			Buffer tmpBuffer = inputBuffer;
			inputBuffer = outputBuffer;
			outputBuffer = tmpBuffer;

			if (iter % printfreq == 0) {
				if (!checkerr) {
					printf("Completed iteration %d\n", iter);
				} else {
					printf("Completed iteration %d, error = %g\n", iter, error);
				}
			}
		}

		if (iter > numiter) {
			iter = numiter;
		}

		tstop = gettime();

		ttot = tstop - tstart;
		titer = ttot / (double)iter;

		printf("\n... finished\n");
		printf("After %d iterations, the error is %g\n", iter, error);
		printf("Time for %d iterations was %g seconds\n", iter, ttot);
		printf("Each iteration took %g seconds\n", titer);

		//output results
		//writedatafiles(psi,m,n, scalefactor);
		//writeplotfile(m,n,scalefactor);

		free(psi);
		free(psitmp);
		printf("... finished\n");

		return 0;
	} catch (Error error) {
		std::cout << error.what() << "(" << error.err() << ")" << std::endl;
		std::cout << "Build Status: " << program.getBuildInfo<CL_PROGRAM_BUILD_STATUS>(cl::Device::getDefault()) << std::endl;
		std::cout << "Build Options:\t" << program.getBuildInfo<CL_PROGRAM_BUILD_OPTIONS>(cl::Device::getDefault()) << std::endl;
		std::cout << "Build Log:\t " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(cl::Device::getDefault()) << std::endl;
	}

	return 0;
}
