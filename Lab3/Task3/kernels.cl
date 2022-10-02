kernel void task3(global double *inputBuffer, global double *outputBuffer, const int m, const int n) {
	int i = get_global_id(0) + 1;
    int j = get_global_id(1) + 1;
	
	outputBuffer[i * (m + 2) + j] = 0.25 * (inputBuffer[(i - 1) * (m + 2) + j] + inputBuffer[(i + 1) * (m + 2) + j] + inputBuffer[i * (m + 2) + j - 1] + inputBuffer[i * (m + 2) + j + 1]);
}