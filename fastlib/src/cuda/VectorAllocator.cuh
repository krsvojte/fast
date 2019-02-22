#include "CudaUtility.h"

template <typename T>
struct VectorAllocator {

private:
	struct Slot {
		bool onDevice;
		int index;
		int currentItem;
	};

public:

	VectorAllocator() {
		cudaStreamCreate(&streamToDevice);
		cudaStreamCreate(&streamToHost);
		_dataDevice = nullptr;
		_dataHost = nullptr;
	}

	~VectorAllocator() {
		free();
	}

	bool alloc(
		size_t vecN,
		size_t vecMinDeviceN,
		size_t N,
		size_t swapIndex = 0
	) {


		size_t totalNeededMemory = N * sizeof(T) * vecN;
		size_t free, total;
		cudaMemGetInfo(&free, &total);

		_swapIndex = swapIndex;
		_singleVecSize = N * sizeof(T);

		printf("Free GPU memory: %f GB\n", (free) / (1024.0f*1024.0f*1024.0f));

		size_t availableVectors = free / (N * sizeof(T));
		if (availableVectors > vecN)
			availableVectors = vecN;

		if (availableVectors < vecMinDeviceN)
			return false;

		printf("Allocating GPU memory: %f GB\n", (availableVectors * N * sizeof(T)) / (1024.0f*1024.0f*1024.0f));
		_CUDA(cudaMalloc(&_dataDevice, availableVectors * N * sizeof(T)));

		if (free >= totalNeededMemory) {
			_deviceVecN = vecN;
			_hostVecN = 0;
		}
		else {
			_deviceVecN = availableVectors;
			_hostVecN = (vecN - availableVectors);

			size_t alignment = 4 * 1024;
			size_t hostSizeAligned = ((_hostVecN * N * sizeof(T) + alignment - 1) / alignment) * alignment;
			printf("Allocating pinned memory: %f GB\n", (hostSizeAligned) / (1024.0f*1024.0f*1024.0f));



			_dataHost = new unsigned char[hostSizeAligned];
			_CUDA(cudaHostRegister(_dataHost, hostSizeAligned, cudaHostRegisterPortable));

		}

		_deviceSlots.resize(_deviceVecN);
		for (auto i = 0; i < _deviceVecN; i++) {
			_deviceSlots[i].index = i;
			_deviceSlots[i].onDevice = true;
		}


		_hostSlots.resize(_hostVecN);
		for (auto i = 0; i < _hostVecN; i++) {
			_hostSlots[i].index = i;
			_hostSlots[i].onDevice = false;
		}

		_itemToSlot.resize(vecN);
		for (auto i = 0; i < vecN; i++) {
			if (i < _deviceVecN) {
				_itemToSlot[i] = &_deviceSlots[i];
				_deviceSlots[i].currentItem = i;
			}
			else {
				_itemToSlot[i] = &_hostSlots[i - _deviceVecN];
				_hostSlots[i - _deviceVecN].currentItem = i;
			}

		}

		return true;
	}

	void free() {
		if (_dataDevice) {
			_CUDA(cudaFree(_dataDevice));
			_dataDevice = nullptr;
		}

		if (_dataHost) {
			_CUDA(cudaHostUnregister(_dataHost));
			delete[] _dataHost;
			_dataHost = nullptr;
		}

	}

	void request() {

	}

	void require() {

	}


	T * get(int itemIndex)
	{
		//if(_itemToSlot_device[index])
		const auto & slot = *_itemToSlot[itemIndex];
		if (slot.onDevice) {
			return reinterpret_cast<T*>(((unsigned char*)_dataDevice) + (_singleVecSize * slot.index));
		}
		else {

		}
		return nullptr;
	}

	void discard(int index) {

	}

private:

	void syncCopy() {

	}


	void * _dataDevice = nullptr;
	void * _dataHost = nullptr;
	size_t _deviceVecN;
	size_t _hostVecN;

	size_t _swapIndex;
	size_t _singleVecSize;

	cudaStream_t streamToDevice;
	cudaStream_t streamToHost;



	std::vector<Slot> _deviceSlots;
	std::vector<Slot> _hostSlots;

	std::vector<Slot*> _itemToSlot;

	/*std::vector<int> _slotToItem_device;
	std::vector<int> _itemToSlot_device;

	std::vector<int> _slotToItem_host;
	std::vector<int> _itemToSlot_host;*/
};
