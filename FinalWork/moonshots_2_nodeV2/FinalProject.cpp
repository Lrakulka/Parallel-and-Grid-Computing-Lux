// FinalProject.cpp : Defines the entry point for the console application.
// g++ FinalProject.cpp -o FinalProject -pthread -std=gnu++0x -D_GLIBCXX_USE_NAN$

#include <algorithm>
#include <string>
#include <iostream>
#include <thread>
#include <ctime>
#include <cmath>
#include <fstream>  

using namespace std;

/* Green operations*/
// First Level.
void loadAdd(volatile long *load, volatile long *param);

void loadSub(volatile long *load, volatile long *param);

void loadCom(volatile long *load, volatile long *param);

// Second level
void loadAbs(volatile long *load, volatile long *param);

// Third level
void loadMul(volatile long *load, volatile long *param);

// Special test of multiplication and shifting
void loadMulBy2(volatile long *load, volatile long *param);

void loadShiftBy2(volatile long *load, volatile long *param);

/* Orange operations*/
void loadDivis(volatile long *load, volatile long *param);

void loadMod(volatile long *load, volatile long *param);

/* Red operations*/
void loadExp(volatile long *load, volatile long *param);

void loadSQRT(volatile long *load, volatile long *param);

void loadSin(volatile long *load, volatile long *param);

void loadAsin(volatile long *load, volatile long *param);

void loadPow(volatile long *load, volatile long *param);

void loadDrop(volatile long *load, volatile long *param);

void loadNotSortedArray(volatile long *load, volatile long *param);

void loadSortedArray(volatile long *load, volatile long *param);

/*
No point in this test
void loadSmoothDrop(volatile long *load, volatile long *param);
*/

void stopThreads(volatile bool *flag, int sleepTime);

void printThreadData(unsigned, double*, unsigned long long*, string);

// Load system with Math operstions
void runTestOnThreads(double *time, unsigned long long* iteration, void(*loadFunction)(volatile long *, volatile long *));

void test(void(*loadFunction)(volatile long *, volatile long *), unsigned int, unsigned int, string);

#define TIME_BETWEEN_TESTS 10
#define TIME_FOR_TESTS 60

volatile bool testStopFlag;
thread* t;
double* threadsTime;
unsigned long long* thredsIterations;
std::ofstream file;

int main(int argc, char* argv[]) {
	int runTest;
	if (argc < 2) {
		runTest = -1;
	}
	else {
		runTest = stoi(argv[1]);
	}
	//may return 0 when not able to detect
	unsigned threadsNumb = std::thread::hardware_concurrency();

	if (!threadsNumb) {
		cout << "Can not  detect number of threads" << endl;
		return 3;
	}

	time_t tim = time(0);   // get time now
	struct tm now;
	now = *localtime(&tim);
	file.open("testLog_" + to_string(now.tm_mday) + "^"
		+ to_string(now.tm_hour) + "^" + to_string(now.tm_min)
		+ "-N" + to_string(runTest) + ".txt");
	if (!file.is_open()) {
		cout << "Can not  open file" << endl;
		return 3;
	}
	t = new thread[threadsNumb - 1];
	threadsTime = new double[threadsNumb];
	thredsIterations = new unsigned long long[threadsNumb];

	file << "Number of available threads " << threadsNumb << endl;
	switch (runTest)
	{
	case 1:
		test(&loadAdd, TIME_FOR_TESTS, threadsNumb, "Addition");
		break;
	case 2:
		test(&loadSub, TIME_FOR_TESTS, threadsNumb, "Subtraction");
		break;
	case 3:
		test(&loadCom, TIME_FOR_TESTS, threadsNumb, "Comparesing");
		break;
	case 4:
		test(&loadDivis, TIME_FOR_TESTS, threadsNumb, "Division");
		break;
	case 5:
		test(&loadMod, TIME_FOR_TESTS, threadsNumb, "Modulus");
		break;
	case 6:
		test(&loadExp, TIME_FOR_TESTS, threadsNumb, "Exponenta");
		break;
	case 7:
		test(&loadSQRT, TIME_FOR_TESTS, threadsNumb, "SQRT");
		break;
	case 8:
		test(&loadSin, TIME_FOR_TESTS, threadsNumb, "Sinus");
		break;
	case 9:
		test(&loadAsin, TIME_FOR_TESTS, threadsNumb, "Asin");
		break;
	case 10:
		test(&loadPow, TIME_FOR_TESTS, threadsNumb, "Pow");
		break;
	case 11:
		// Special tests
		test(&loadMulBy2, TIME_FOR_TESTS, threadsNumb, "Multiplication by 2");
		break;
	case 12:
		test(&loadShiftBy2, TIME_FOR_TESTS, threadsNumb, "Multiplication by 2 but shifting uses");
		break;
	case 13:
		// Test drop behewior
		test(&loadDrop, TIME_FOR_TESTS, threadsNumb, "Sudden drops");
		break;
	case 14:
		// Why is it faster to process a sorted array than an unsorted array?
		test(&loadNotSortedArray, TIME_FOR_TESTS, threadsNumb, "Not sorted Array");
		break;
	case 15:
		test(&loadSortedArray, TIME_FOR_TESTS, threadsNumb, "Sorted Array");
		break;
	default:
		// Green operations
		test(&loadAdd, TIME_FOR_TESTS, threadsNumb, "Addition");
		test(&loadSub, TIME_FOR_TESTS, threadsNumb, "Subtraction");
		test(&loadCom, TIME_FOR_TESTS, threadsNumb, "Comparesing");
		// Orange operations
		test(&loadDivis, TIME_FOR_TESTS, threadsNumb, "Division");
		test(&loadMod, TIME_FOR_TESTS, threadsNumb, "Modulus");
		// Red operations
		test(&loadExp, TIME_FOR_TESTS, threadsNumb, "Exponenta");
		test(&loadSQRT, TIME_FOR_TESTS, threadsNumb, "SQRT");
		test(&loadSin, TIME_FOR_TESTS, threadsNumb, "Sinus");
		test(&loadAsin, TIME_FOR_TESTS, threadsNumb, "Asin");
		test(&loadPow, TIME_FOR_TESTS, threadsNumb, "Pow");
		// Special tests
		test(&loadMulBy2, TIME_FOR_TESTS, threadsNumb, "Multiplication by 2");
		test(&loadShiftBy2, TIME_FOR_TESTS, threadsNumb, "Multiplication by 2 but shifting uses");
		// Test drop behewior
		test(&loadDrop, TIME_FOR_TESTS, threadsNumb, "Sudden drops");
		// Why is it faster to process a sorted array than an unsorted array?
		test(&loadNotSortedArray, TIME_FOR_TESTS, threadsNumb, "Not sorted Array");
		test(&loadSortedArray, TIME_FOR_TESTS, threadsNumb, "Sorted Array");
		break;
	}

	file.close();
	delete[] t, threadsTime, thredsIterations;
	return 0;
}

void test(void(*loadFunction)(volatile long *, volatile long *), unsigned int testTime, unsigned int threadsNumb, string testName) {
	testStopFlag = true;
	for (int i = 0; i < threadsNumb - 1; ++i) {
		// Constructs the new thread and runs it. Does not block execution.
		t[i] = thread(runTestOnThreads, &threadsTime[i], &thredsIterations[i], loadFunction);
	}
	thread stopThread(stopThreads, &testStopFlag, testTime);
	runTestOnThreads(&threadsTime[threadsNumb - 1], &thredsIterations[threadsNumb - 1], loadFunction);
	stopThread.join();
	for (int i = 0; i < threadsNumb - 1; ++i) {
		t[i].join();
	}
	printThreadData(threadsNumb, threadsTime, thredsIterations, testName);
	// Make distance of mesuremence
	this_thread::sleep_for(chrono::seconds(TIME_BETWEEN_TESTS));
}

void printThreadData(unsigned threadsNumb, double* times, unsigned long long* iterations, string testName) {
	file << "Test name is " << testName << endl;
	file << "Thread \t Time \t Iteration made " << endl;
	unsigned long long min, max, aver;
	min = iterations[0];
	max = aver = 0;
	for (int i = 0; i < threadsNumb; ++i) {
		file << i << " \t " << times[i] << " \t " << iterations[i] << endl;
		if (min > iterations[i]) {
			min = iterations[i];
		}
		if (max < iterations[i]) {
			max = iterations[i];
		}
		aver += iterations[i];
	}
	file << "min \t max \t average " << endl;
	file << min << " \t\t " << max << " \t\t " << (aver / threadsNumb) << endl;
}

void stopThreads(volatile bool *flag, int sleepTime) {
	this_thread::sleep_for(chrono::seconds(sleepTime));
	*flag = false;
}

// Load system with Math 
void runTestOnThreads(double *time, unsigned long long* iteration, void(*loadFunction)(volatile long *, volatile long *))
{
	clock_t begin = clock();
	*iteration = 0;
	// Protection for variable optimization
	volatile long load = 2;
	volatile long param = 0;
	while (testStopFlag) {
		(*iteration)++;
		loadFunction(&load, &param);
	}
	*time = double(clock() - begin) / CLOCKS_PER_SEC;
}

/* Green operations*/
// First Level.
void loadAdd(volatile long *load, volatile long *param) {
	(*load)++;
}

void loadSub(volatile long *load, volatile long *param) {
	(*load)--;
}

void loadCom(volatile long *load, volatile long *param) {
	(*load) == (*load);
}

// Second level
void loadAbs(volatile long *load, volatile long *param) {
	*load = abs(*load);
}

// Third level
void loadMul(volatile long *load, volatile long *param) {
	*load = *load * *load;
}

// Special test of multiplication and shifting
void loadMulBy2(volatile long *load, volatile long *param) {
	*load = *load * 2;
}

void loadShiftBy2(volatile long *load, volatile long *param) {
	*load = *load << 1;
}

/* Orange operations*/
void loadDivis(volatile long *load, volatile long *param) {
	*load = *load / 3;
}

void loadMod(volatile long *load, volatile long *param) {
	*load = *load % 3;
}

/* Red operations*/
void loadExp(volatile long *load, volatile long *param) {
	*load = exp(*load);
}

void loadSQRT(volatile long *load, volatile long *param) {
	*load = sqrt(*load);
}

void loadSin(volatile long *load, volatile long *param) {
	*load = sin(*load);
}

void loadAsin(volatile long *load, volatile long *param) {
	*load = asin(*load);
}

void loadPow(volatile long *load, volatile long *param) {
	*load = pow(*load, 3);
}

void loadDrop(volatile long *load, volatile long *param) {
	(*load)++;
	if (*load % 50000000 == 0) {
		this_thread::sleep_for(chrono::seconds(4));
	}
}
/*
No point in this test
void loadSmoothDrop(volatile long *load, volatile long *param) {
(*load)++;
this_thread::sleep_for(std::chrono::milliseconds(abs(*param)));
if (*load % 100000 == 0) {
(*param)++;
}
if (*param == 4000) {
*param *= -1;
}
} */

void loadSortedArray(volatile long *load, volatile long *param) {
	// Generate data
	const unsigned arraySize = 32768;
	int data[arraySize];

	for (unsigned c = 0; c < arraySize; ++c)
		data[c] = rand() % 256;

	// !!! With this, the next loop runs faster
	sort(data, data + arraySize);

	volatile long long sum = 0;

	for (unsigned i = 0; i < 100000; ++i)
	{
		// Primary loop
		for (unsigned c = 0; c < arraySize; ++c)
		{
			if (data[c] >= 128)
				sum += data[c];
		}
	}
}

void loadNotSortedArray(volatile long *load, volatile long *param) {
	// Generate data
	const unsigned arraySize = 32768;
	int data[arraySize];

	for (unsigned c = 0; c < arraySize; ++c)
		data[c] = rand() % 256;


	volatile long long sum = 0;

	for (unsigned i = 0; i < 100000; ++i)
	{
		// Primary loop
		for (unsigned c = 0; c < arraySize; ++c)
		{
			if (data[c] >= 128)
				sum += data[c];
		}
	}
}