#define L 512

kernel void task2(global double	*sequence, const int N) {
	int localId = get_local_id(0);
	int localSize = get_local_size(0);
	int groupId = get_group_id(0);

	int globalSize = get_global_size(0);
	int numberOfElements = N / globalSize;

	if (numberOfElements <= 0) {
		numberOfElements = 1;
	}

	int start = get_global_id(0) * numberOfElements;
	int end = start + numberOfElements;

	local double localSumArray[L];

	if (end <= N) { 
		double h = 1.0 / (double) N;
		double sum = 0;

		for (int i = start; i < end; i++) {
			double x = h * ((double) i - 0.5);
			sum += 4.0 / (1.0 + x * x);
		}

		sum *= h;

		localSumArray[localId] = sum;
		barrier(CLK_LOCAL_MEM_FENCE);

		// Reduce localSumArray to totalSum

		for (int i = 0; (i < log2((double) L)) && (localId % ((int) pow(2.0, (double) i)) == 0); i++) {
			int destId = localId ^ (int) pow(2.0, (double) i);

			if (localId % (int) pow(2.0, (double) i + 1)) {
				localSumArray[destId] += localSumArray[localId];
			}

			barrier(CLK_LOCAL_MEM_FENCE);
		}

		double totalSum = localSumArray[0];

		if (localId == 0) {
			sequence[groupId] = totalSum;
		}
	}
}