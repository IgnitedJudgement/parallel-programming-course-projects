int isPrime(int number);
int countPrimes(int start, int end, int *sequenceBuffer);

kernel void task1(global const int *sequence, global int *nPrimesBuffer, const int N) {
	int globalSize = get_global_size(0);
	int numberOfElements = N / globalSize;

	if (numberOfElements <= 0) {
		numberOfElements = 1;
	}

	int start = get_global_id(0) * numberOfElements;
	int end = start + numberOfElements;

	if (end <= N) { 
		atomic_add(&nPrimesBuffer[0], countPrimes(start, end, sequence));
	}
}

int isPrime(int number) {
	if (number <= 1) {
		return 0;
	}

	for (int i = 2; i <= sqrt((double) number); i++) {
		if (number % i == 0) {
			return 0;
		}
	}

	return 1;
}

int countPrimes(int start, int end, int *sequence) {
	int counter = 0;

	for (int i = start; i < end; i++) {
		if (isPrime(sequence[i]) == 1) {
			counter++;
		}
	}

	return counter;
}