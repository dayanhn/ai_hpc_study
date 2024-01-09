#include "util_func.h"

void show_device_prop(){
	cudaDeviceProp prop;
	int dev_count = 0;
	CUDA_CHECK_RETURN(cudaGetDeviceCount(&dev_count));
	cout <<"find "<< dev_count << " gpu card!\n";
	
	for(int i = 0; i < dev_count; i++){
		CUDA_CHECK_RETURN(cudaGetDeviceProperties(&prop,i));
		cout <<"\ndevice " << i << " properties:" << endl;
		cout <<   "name[256]:                  " << prop.name << endl;                  /**< ASCII string identifying device */
		cout <<   "totalGlobalMem:             " << prop.totalGlobalMem / (1024.0 * 1024 * 1024) << "G" << endl;             /**< Global memory available on device in bytes */
		cout <<   "sharedMemPerBlock:          " << prop.sharedMemPerBlock << endl;          /**< Shared memory available per block in bytes */
		cout <<   "regsPerBlock:               " << prop.regsPerBlock << endl;               /**< 32-bit registers available per block */
		cout <<   "warpSize:                   " << prop.warpSize << endl;                   /**< Warp size in threads */
		cout <<   "memPitch:                   " << prop.memPitch / (1024.0 * 1024 * 1024) << "G" << endl;                   /**< Maximum pitch in bytes allowed by memory copies */
		cout <<   "maxThreadsPerBlock:         " << prop.maxThreadsPerBlock << endl;         /**< Maximum number of threads per block */
		cout <<   "maxThreadsDim[3]:           " << "[" << prop.maxThreadsDim[0] << ","\
				<< prop.maxThreadsDim[1] << "," << prop.maxThreadsDim[2] <<"]"<< endl;           /**< Maximum size of each dimension of a block */
		cout <<   "maxGridSize[3]:             " << "[" << prop.maxGridSize[0] << ","\
				<< prop.maxGridSize[1]   << "," << prop.maxGridSize[2]   <<"]"<< endl;             /**< Maximum size of each dimension of a grid */
		cout <<   "clockRate:                  " << prop.clockRate / (1000.0 * 1000) << "G" << endl;                  /**< Clock frequency in kilohertz */
		cout <<   "totalConstMem:              " << prop.totalConstMem << endl;              /**< Constant memory available on device in bytes */
		cout <<   "major:                      " << prop.major << endl;                      /**< Major compute capability */
		cout <<   "minor:                      " << prop.minor << endl;                      /**< Minor compute capability */
		cout <<   "textureAlignment:           " << prop.textureAlignment << endl;           /**< Alignment requirement for textures */
		cout <<   "texturePitchAlignment:      " << prop.texturePitchAlignment << endl;      /**< Pitch alignment requirement for texture references bound to pitched memory */
		cout <<   "deviceOverlap:              " << prop.deviceOverlap << endl;              /**< Device can concurrently copy memory and execute a kernel. Deprecated. Use instead asyncEngineCount. */
		cout <<   "multiProcessorCount:        " << prop.multiProcessorCount << endl;        /**< Number of multiprocessors on device */
		cout <<   "kernelExecTimeoutEnabled:   " << prop.kernelExecTimeoutEnabled << endl;   /**< Specified whether there is a run time limit on kernels */
		cout <<   "integrated:                 " << prop.integrated << endl;                 /**< Device is integrated as opposed to discrete */
		cout <<   "canMapHostMemory:           " << prop.canMapHostMemory << endl;           /**< Device can map host memory with cudaHostAlloc/cudaHostGetDevicePointer */
		cout <<   "computeMode:                " << prop.computeMode << endl;                /**< Compute mode (See ::cudaComputeMode) */
		cout <<   "maxTexture1D:               " << prop.maxTexture1D << endl;               /**< Maximum 1D texture size */
		cout <<   "maxTexture1DMipmap:         " << prop.maxTexture1DMipmap << endl;         /**< Maximum 1D mipmapped texture size */
		cout <<   "maxTexture1DLinear:         " << prop.maxTexture1DLinear << endl;         /**< Maximum size for 1D textures bound to linear memory */
		printf("maxTexture2D[2]:            [%d,%d]\n" ,prop.maxTexture2D[0],prop.maxTexture2D[1] );            /**< Maximum 2D texture dimensions */
		printf("maxTexture2DMipmap[2]:      [%d,%d]\n" ,prop.maxTexture2DMipmap[0],prop.maxTexture2DMipmap[1] );      /**< Maximum 2D mipmapped texture dimensions */
		printf("maxTexture2DLinear[3]:      [%d,%d,%d]\n" ,prop.maxTexture2DLinear[0],prop.maxTexture2DLinear[1],prop.maxTexture2DLinear[2] );      /**< Maximum dimensions (width, height, pitch) for 2D textures bound to pitched memory */
		printf("maxTexture2DGather[2]:      [%d,%d]\n" ,prop.maxTexture2DGather[0],prop.maxTexture2DGather[1] );      /**< Maximum 2D texture dimensions if texture gather operations have to be performed */
		printf("maxTexture3D[3]:            [%d,%d,%d]\n" ,prop.maxTexture3D[0],prop.maxTexture3D[1],prop.maxTexture3D[2] );            /**< Maximum 3D texture dimensions */
		printf("maxTexture3DAlt[3]:         [%d,%d,%d]\n" ,prop.maxTexture3DAlt[0],prop.maxTexture3DAlt[1],prop.maxTexture3DAlt[2] );         /**< Maximum alternate 3D texture dimensions */
		printf("maxTextureCubemap:          %d\n" ,prop.maxTextureCubemap );          /**< Maximum Cubemap texture dimensions */
		printf("maxTexture1DLayered[2]:     [%d,%d]\n" ,prop.maxTexture1DLayered[0],prop.maxTexture1DLayered[1] );     /**< Maximum 1D layered texture dimensions */
		printf("maxTexture2DLayered[3]:     [%d,%d,%d]\n" ,prop.maxTexture2DLayered[0],prop.maxTexture2DLayered[1],prop.maxTexture2DLayered[2] );     /**< Maximum 2D layered texture dimensions */
		printf("maxTextureCubemapLayered[2]:[%d,%d]\n" ,prop.maxTextureCubemapLayered[0],prop.maxTextureCubemapLayered[1] );/**< Maximum Cubemap layered texture dimensions */
		printf("maxSurface1D:               %d\n" ,prop.maxSurface1D );               /**< Maximum 1D surface size */
		printf("maxSurface2D[2]:            [%d,%d]\n" ,prop.maxSurface2D[0],prop.maxSurface2D[1] );            /**< Maximum 2D surface dimensions */
		printf("maxSurface3D[3]:            [%d,%d,%d]\n" ,prop.maxSurface3D[0],prop.maxSurface3D[1],prop.maxSurface3D[2] );            /**< Maximum 3D surface dimensions */
		printf("maxSurface1DLayered[2]:     [%d,%d]\n" ,prop.maxSurface1DLayered[0],prop.maxSurface1DLayered[1] );     /**< Maximum 1D layered surface dimensions */
		printf("maxSurface2DLayered[3]:     [%d,%d,%d]\n" ,prop.maxSurface2DLayered[0],prop.maxSurface2DLayered[1],prop.maxSurface2DLayered[2] );     /**< Maximum 2D layered surface dimensions */
		printf("maxSurfaceCubemap:          %d\n" ,prop.maxSurfaceCubemap );          /**< Maximum Cubemap surface dimensions */
		printf("maxSurfaceCubemapLayered[2]:[%d,%d]\n" ,prop.maxSurfaceCubemapLayered[0],prop.maxSurfaceCubemapLayered[1] );/**< Maximum Cubemap layered surface dimensions */		cout <<   "surfaceAlignment:           " << prop.surfaceAlignment << endl;           /**< Alignment requirements for surfaces */
		cout <<   "concurrentKernels:          " << prop.concurrentKernels << endl;          /**< Device can possibly execute multiple kernels concurrently */
		cout <<   "ECCEnabled:                 " << prop.ECCEnabled << endl;                 /**< Device has ECC support enabled */
		cout <<   "pciBusID:                   " << prop.pciBusID << endl;                   /**< PCI bus ID of the device */
		cout <<   "pciDeviceID:                " << prop.pciDeviceID << endl;                /**< PCI device ID of the device */
		cout <<   "pciDomainID:                " << prop.pciDomainID << endl;                /**< PCI domain ID of the device */
		cout <<   "tccDriver:                  " << prop.tccDriver << endl;                  /**< 1 if device is a Tesla device using TCC driver, 0 otherwise */
		cout <<   "asyncEngineCount:           " << prop.asyncEngineCount << endl;           /**< Number of asynchronous engines */
		cout <<   "unifiedAddressing:          " << prop.unifiedAddressing << endl;          /**< Device shares a unified address space with the host */
		cout <<   "memoryClockRate:            " << prop.memoryClockRate << endl;            /**< Peak memory clock frequency in kilohertz */
		cout <<   "memoryBusWidth:             " << prop.memoryBusWidth << endl;             /**< Global memory bus width in bits */
		cout <<   "l2CacheSize:                " << prop.l2CacheSize << endl;                /**< Size of L2 cache in bytes */
		cout <<   "maxThreadsPerMultiProcessor:" << prop.maxThreadsPerMultiProcessor << endl;/**< Maximum resident threads per multiprocessor */
		cout <<   "streamPrioritiesSupported:  " << prop.streamPrioritiesSupported << endl;  /**< Device supports stream priorities */
		cout <<   "globalL1CacheSupported:     " << prop.globalL1CacheSupported << endl;     /**< Device supports caching globals in L1 */
		cout <<   "localL1CacheSupported:      " << prop.localL1CacheSupported << endl;      /**< Device supports caching locals in L1 */
		cout <<   "sharedMemPerMultiprocessor: " << prop.sharedMemPerMultiprocessor << endl; /**< Shared memory available per multiprocessor in bytes */
		cout <<   "regsPerMultiprocessor:      " << prop.regsPerMultiprocessor << endl;      /**< 32-bit registers available per multiprocessor */
		cout <<   "managedMemory:              " << prop.managedMemory << endl;              /**< Device supports allocating managed memory on this system */
		cout <<   "isMultiGpuBoard:            " << prop.isMultiGpuBoard << endl;            /**< Device is on a multi-GPU board */
		cout <<   "multiGpuBoardGroupID:       " << prop.multiGpuBoardGroupID << endl;       /**< Unique identifier for a group of devices on the same multi-GPU board */
		cout <<   "hostNativeAtomicSupported:  " << prop.hostNativeAtomicSupported << endl;  /**< Link between the device and the host supports native atomic operations */
		cout <<   "singleToDoublePrecisionPerfRatio: " << prop.singleToDoublePrecisionPerfRatio << endl;/**< Ratio of single precision performance (in floating-point operations per second) to double precision performance */
		cout <<   "pageableMemoryAccess:       " << prop.pageableMemoryAccess << endl;       /**< Device supports coherently accessing pageable memory without calling cudaHostRegister on it */
		cout <<   "concurrentManagedAccess:    " << prop.concurrentManagedAccess << endl;    /**< Device can coherently access managed memory concurrently with the CPU */
		cout <<   "computePreemptionSupported: " << prop.computePreemptionSupported << endl; /**< Device supports Compute Preemption */
		cout <<   "canUseHostPointerForRegisteredMem: " << prop.canUseHostPointerForRegisteredMem << endl; /**< Device can access host registered memory at the same virtual address as the CPU */
		cout <<   "cooperativeLaunch:          " << prop.cooperativeLaunch << endl; /**< Device supports launching cooperative kernels via ::cudaLaunchCooperativeKernel */
		cout <<   "cooperativeMultiDeviceLaunch: " << prop.cooperativeMultiDeviceLaunch << endl; /**< Device can participate in cooperative kernels launched via ::cudaLaunchCooperativeKernelMultiDevice */
		cout <<   "sharedMemPerBlockOptin:     " << prop.sharedMemPerBlockOptin << endl; /**< Per device maximum shared memory per block usable by special opt in */
	}
}
