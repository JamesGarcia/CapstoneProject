#include <stdlib.h>
#include <fstream>
#include <iostream>
//#include <boost/filesystem.hpp>

#include "../../capstone_kernels/src/krnl_L1.h"
#include "../../capstone_kernels/src/krnl_L2.h"
#include "../../capstone_kernels/src/krnl_L3.h"
#include "../../capstone_kernels/src/cnn_util.h"


#if PROD

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/opencv_modules.hpp>
#include <opencv2/core/core.hpp>
//#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>
#include <opencv2/videoio/videoio.hpp>

//#include "vadd.h"
#include "dirent.h"

#endif

#include <ctime>
#include <chrono>

#if !PROD
void cnn_top(const cnndata_t *input, const cnndata_t *W1, const cnndata_t *W2, const cnndata_t *W3, cnndata_t *O1, cnndata_t *O2, cnndata_t *O3);
#endif

static const int W1_SIZE = N_1*M_1*K_1*K_1;
static const int W2_SIZE = N_2*M_2*K_2*K_2;
static const int W3_SIZE = N_3*M_3*K_3*K_3;
static const int SCREEN_SIZE = 1080 * 1920;
static const int H1_SIZE = M_1;
static const int H2_SIZE = M_2;

static const int DATA_SIZE = 4096;
static const std::string error_message =
    "Error: Result mismatch:\n"
    "i = %d CPU result = %d Device result = %d\n";

/*

	--- HERE BE DEAD CODE ---


//namespace std_fs = std::filesystem;
namespace boost_fs = boost::filesystem;
static const boost_fs::path root = boost_fs::current_path().root_path();



 **
 * \brief   Return the filenames of all files that have the specified extension
 *          in the specified directory and all subdirectories.
 *
std::vector<boost_fs::path> get_all(boost_fs::path const & root, std::string const & ext)
{
    std::vector<boost_fs::path> paths;
    if (boost_fs::exists(root) && boost_fs::is_directory(root))
    {
        for (auto const & entry : boost_fs::recursive_directory_iterator(root))
        {
            if (boost_fs::is_regular_file(entry) && entry.path().extension() == ext)
                paths.emplace_back(entry.path().filename());
        }
    }

    return paths;
}

*/

int main(int argc, char* argv[]) {

	std::cout << " ========== START OF DEMO ========== " << std::endl;
	std::cout << " Running with PROD = " << PROD << std::endl;

#if PROD
	// Demo Code -- read an mp4 video w/ known name
	// TODO: be able to read any name
	// TODO: read in avi files
	cv::VideoCapture video("/home/root/video.mp4");
	if(!video.open("/home/root/video.mp4")){
		std::cout << "[ERROR] -- video file unopenable" << std::endl;
		return EXIT_FAILURE;
	}

	//TARGET_DEVICE macro needs to be passed from gcc command line
	if(argc != 2) {
		std::cout << "Usage: " << argv[0] <<" <xclbin>" << std::endl;
		return EXIT_FAILURE;
	}

	char* xclbinFilename = argv[1];

	std::vector<cl::Device> devices;
	cl::Device device;
	std::vector<cl::Platform> platforms;
	bool found_device = false;

	//traversing all Platforms To find Xilinx Platform and targeted
	//Device in Xilinx Platform
	cl::Platform::get(&platforms);
	for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
		cl::Platform platform = platforms[i];
		std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
		if ( platformName == "Xilinx"){
			devices.clear();
			platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
		if (devices.size()){
			device = devices[0];
			found_device = true;
			break;
		}
		}
	}
	if (found_device == false){
	   std::cout << "Error: Unable to find Target Device "
		   << device.getInfo<CL_DEVICE_NAME>() << std::endl;
	   return EXIT_FAILURE;
	}else {
		std::cout << "found target device" << std::endl;
	}

	// Creating Context and Command Queue for selected device
	cl::Context context(device);
	cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE);

	// Load xclbin
	std::cout << "Loading: '" << xclbinFilename << "'\n";
	std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
	bin_file.seekg (0, bin_file.end);
	unsigned nb = bin_file.tellg();
	bin_file.seekg (0, bin_file.beg);
	char *buf = new char [nb];
	bin_file.read(buf, nb);

	// Creating Program from Binary File
	std::cout << "Creating Program from Binary File" << std::endl;
	cl::Program::Binaries bins;
	bins.push_back({buf,nb});
	devices.resize(1);
	cl::Program program(context, devices, bins);

	// This call will get the kernel object from program. A kernel is an
	// OpenCL function that is executed on the FPGA.
	std::cout << "Getting kernel object from Program" << std::endl;
	cl::Kernel krnl_srcnn(program,"cnn_top");

	// These commands will allocate memory on the Device. The cl::Buffer objects can
	// be used to reference the memory locations on the device.
	std::cout << "Allocating Memory on Device" << std::endl;
	cl::Buffer buffer_I (context, CL_MEM_READ_ONLY, SCREEN_SIZE*sizeof(cnndata_t));
	std::cout << "Done 1" << std::endl;
	cl::Buffer buffer_W1(context, CL_MEM_READ_ONLY, W1_SIZE*sizeof(cnndata_t));
	std::cout << "Done 2" << std::endl;
	cl::Buffer buffer_W2(context, CL_MEM_READ_ONLY, W2_SIZE*sizeof(cnndata_t));
	std::cout << "Done 3" << std::endl;
	cl::Buffer buffer_W3(context, CL_MEM_READ_ONLY, W3_SIZE*sizeof(cnndata_t));
	std::cout << "Done 4" << std::endl;
	cl::Buffer buffer_O1(context, CL_MEM_READ_WRITE, SCREEN_SIZE*H1_SIZE*sizeof(cnndata_t));
	std::cout << "Done 5" << std::endl;
	cl::Buffer buffer_O2(context, CL_MEM_READ_WRITE, SCREEN_SIZE*H2_SIZE*sizeof(cnndata_t));
	std::cout << "Done 6" << std::endl;
	cl::Buffer buffer_O3(context, CL_MEM_WRITE_ONLY, SCREEN_SIZE*sizeof(cnndata_t));
	std::cout << "Done 7" << std::endl;

	//set the kernel Arguments
	std::cout << "Set Kernel Arguments" << std::endl;
	int narg=0;
	krnl_srcnn.setArg(narg++,buffer_I );
	krnl_srcnn.setArg(narg++,buffer_W1);
	krnl_srcnn.setArg(narg++,buffer_W2);
	krnl_srcnn.setArg(narg++,buffer_W3);
	krnl_srcnn.setArg(narg++,buffer_O1);
	krnl_srcnn.setArg(narg++,buffer_O2);
	krnl_srcnn.setArg(narg++,buffer_O3);

	//We then need to map our OpenCL buffers to get the pointers
	std::cout << "Map OpenCL Buffers to get Pointers" << std::endl;
	float *ptr_I = (float *) q.enqueueMapBuffer (buffer_I , CL_TRUE, CL_MAP_WRITE, 0, SCREEN_SIZE*sizeof(cnndata_t));
	float *ptr_W1= (float *) q.enqueueMapBuffer (buffer_W1, CL_TRUE, CL_MAP_WRITE, 0, W1_SIZE*sizeof(cnndata_t));
	float *ptr_W2= (float *) q.enqueueMapBuffer (buffer_W2, CL_TRUE, CL_MAP_WRITE, 0, W2_SIZE*sizeof(cnndata_t));
	float *ptr_W3= (float *) q.enqueueMapBuffer (buffer_W3, CL_TRUE, CL_MAP_WRITE, 0, W3_SIZE*sizeof(cnndata_t));
	float *ptr_O1= (float *) q.enqueueMapBuffer (buffer_O1, CL_TRUE,  CL_MAP_READ, 0, SCREEN_SIZE*H1_SIZE*sizeof(cnndata_t));
	float *ptr_O2= (float *) q.enqueueMapBuffer (buffer_O2, CL_TRUE,  CL_MAP_READ, 0, SCREEN_SIZE*H2_SIZE*sizeof(cnndata_t));
	float *ptr_O3= (float *) q.enqueueMapBuffer (buffer_O3, CL_TRUE,  CL_MAP_READ, 0, SCREEN_SIZE*sizeof(cnndata_t));

#else
	cnndata_t *ptr_I  = (cnndata_t*) malloc(SCREEN_SIZE*sizeof(cnndata_t));
	cnndata_t *ptr_W1 = (cnndata_t*) malloc(W1_SIZE*sizeof(cnndata_t));
	cnndata_t *ptr_W2 = (cnndata_t*) malloc(W2_SIZE*sizeof(cnndata_t));
	cnndata_t *ptr_W3 = (cnndata_t*) malloc(W3_SIZE*sizeof(cnndata_t));
	cnndata_t *ptr_O1 = (cnndata_t*) malloc(SCREEN_SIZE*H1_SIZE*sizeof(cnndata_t));
	cnndata_t *ptr_O2 = (cnndata_t*) malloc(SCREEN_SIZE*H2_SIZE*sizeof(cnndata_t));
	cnndata_t *ptr_O3 = (cnndata_t*) malloc(SCREEN_SIZE*sizeof(cnndata_t));
	if(!ptr_I) std::cout << "null ptr on input" << std::endl;
	if(!ptr_W1) std::cout << "null ptr on weights1" << std::endl;
	if(!ptr_W2) std::cout << "null ptr on weights2" << std::endl;
	if(!ptr_W3) std::cout << "null ptr on weights3" << std::endl;
	if(!ptr_O1) std::cout << "null ptr on output1" << std::endl;
	if(!ptr_O2) std::cout << "null ptr on output2" << std::endl;
	if(!ptr_O3) std::cout << "null ptr on output3" << std::endl;

#endif

	//setting weights data
	// DEMO: using dummy data
	// TODO: don't use dummy data
	for(int i = 0; i<W1_SIZE; i++){
		ptr_W1[i] = ((100.0) * ((float)rand() / RAND_MAX));
	}
	for(int i = 0; i<W2_SIZE; i++){
		ptr_W2[i] = ((100.0) * ((float)rand() / RAND_MAX));
	}
	for(int i = 0; i<W3_SIZE; i++){
		ptr_W3[i] = ((100.0) * ((float)rand() / RAND_MAX));
	}

	/**
	 * Iterative frame processor 
	 * 
	 * */

	cv::Mat frame;
	do {
		/**
		 * Process frame by frame
		*/
		video >> frame;
		if(frame.empty()){
			std::cout << "[ERROR] -- video file generated blank frame" << std::endl;
		}

		if(frame.rows != 1080 || frame.cols != 1920){
			std::cout << "[ERROR] -- bad dimensions from video frame" << std::endl;
		}

		//setting input data
		for(int row = 0; row<1080; row++){
			for(int col = 0; col<1920; col++){
#if PROD
				ptr_I[row*1920 + col] = frame.at<float>(row, col);
#else
				ptr_I[row*1920 + col] = rand();
#endif
			}
		}

#if PROD
		// flushes the queue
		q.finish();
#endif

		std::cout << " ========== START KERNEL ========== " << std::endl;
		auto startKernel = std::chrono::high_resolution_clock::now();
#if PROD
		// Data will be migrated to kernel space
		q.enqueueMigrateMemObjects({buffer_I,buffer_W1,buffer_W2,buffer_W3,buffer_O1,buffer_O2},0/* 0 means from host*/);
		// this will apply the cnn kernel
		q.enqueueTask(krnl_srcnn);
		// The result of the previous kernel execution will need to be retrieved in
		// order to view the results. This call will transfer the data from FPGA to
		// source_results vector
		q.enqueueMigrateMemObjects({buffer_O1,buffer_O2,buffer_O3},CL_MIGRATE_MEM_OBJECT_HOST);
		// flushes the queue
		q.finish();
#else
		cnn_top(ptr_I, ptr_W1, ptr_W2, ptr_W3, ptr_O1, ptr_O2, ptr_O3);
#endif
		auto finishKernel = std::chrono::high_resolution_clock::now();
		std::cout << " ========== FINISH KERNEL ========== " << std::endl;
		auto timeKernel = finishKernel - startKernel;
		auto timeKernelMilisecs = std::chrono::duration_cast<std::chrono::microseconds>(timeKernel).count();
		std::cout << " Kernel Latency (us) > " << timeKernelMilisecs << std::endl;
		std::cout << " Kernel throughput (MHz) > " << 1.0 / timeKernelMilisecs << std::endl;
		std::cout << " ========== START WRITEBACK ========== " << std::endl;

#if PROD
		//write back data
		for(int row = 0; row<1080; row++){
			for(int col = 0; col<1920; col++){
				frame.at<float>(row, col) = ptr_O3[row*1920 + col];
			}
		}
#else
		std::cout << "No writeback in Test" << std::endl;
#endif

		std::cout << " ========== FINISH WRITEBACK ========== " << std::endl;



		//Verify the result
		// not doing this for demo
		// TODO: verify the output of the CNN check res

		/*
		int match = 0;
		for (int i = 0; i < DATA_SIZE; i++) {
			int host_result = ptr_a[i] + ptr_b[i];
			if (ptr_result[i] != host_result) {
				printf(error_message.c_str(), i, host_result, ptr_result[i]);
				match = 1;
				break;
			}
		}*/
#if PROD
		q.enqueueUnmapMemObject(buffer_I  , ptr_I );
		q.enqueueUnmapMemObject(buffer_W1 , ptr_W1);
		q.enqueueUnmapMemObject(buffer_W2 , ptr_W2);
		q.enqueueUnmapMemObject(buffer_W3 , ptr_W3);
		q.enqueueUnmapMemObject(buffer_O1 , ptr_O1);
		q.enqueueUnmapMemObject(buffer_O2 , ptr_O2);
		q.enqueueUnmapMemObject(buffer_O3 , ptr_O3);
		q.finish();
#else
		free(ptr_I );
		free(ptr_W1);
		free(ptr_W2);
		free(ptr_W3);
		free(ptr_O1);
		free(ptr_O2);
		free(ptr_O3);
#endif

		// no correctness checking in demo code
		/*
		std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
		return (match ? EXIT_FAILURE :  EXIT_SUCCESS);
		*/
#if PROD
		/**
		 * TODO: stream each frame out to video port
		 * */

		cv::VideoWriter writer;
		int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
		writer.open("/home/root/video_out.avi", codec, 30.0, frame.size(), frame.type()==CV_8UC3);
		if(!writer.isOpened()){
			std::cerr << "couldnt open writer" << std::endl;
			return 1;
		}

		//this should write a single framed video out to the file system
		writer.write(frame);
#endif

		std::cout << " ========== END OF DEMO ========== " << std::endl;

	}while(frame != NULL);

	return 0;

	//Verify the result
	// not doing this for demo
	// TODO: this

	/*
	int match = 0;
	for (int i = 0; i < DATA_SIZE; i++) {
		int host_result = ptr_a[i] + ptr_b[i];
		if (ptr_result[i] != host_result) {
			printf(error_message.c_str(), i, host_result, ptr_result[i]);
			match = 1;
			break;
		}
	}*/




	/* Kunal's code was/is buggy A.F. -- makes more sense to rewrite from scratch
    //Get avi files should be a vector with length 1
    std::string ext(".avi");
    std::vector<boost_fs::path> fname;
    boost_fs::path root("/");
    boost_fs::path file;

    // Find avi file
    fname = get_all(root, ext);
    file = fname.back();
    fname.pop_back();

    // capture frames and store in fifo
    std::vector<cv::Mat> frames;

    // float vector representing the pixel values
    std::vector<float *> frame_data;
    cv::Mat frame;
    cv::VideoCapture cap (file);

    // open the default camera via the default api
    // auto-detect the default cv2 api
    int deviceID = 0;
    int apiID = cv::CAP_ANY;

    // open video capture stream to file
    cap.open(deviceID, apiID);

    if(!cap.isOpened()) {
        cerr << "Error! Unable To Open Camera";
    }
    int total_frames = cap.get(CAP_PROP_FRAME_COUNT) - 1;
    int frame_count = 0;

    while(cap.isOpened()) {
        cap.read(frame);
        if(frame.empty()) {
            cerr << "ERROR! blank frame grabbed\n";
            break;
        }
        // Collect frames in fifo
        frames.push_back(frame);
        float *mat_data = (float *)frame.data;
        frame_data.push_back(mat_data);
        frame_count++;

        if(frame_count > (total_frames-1)) {
            cap.release();
            cout << "Done extracting frames...\n";
            cout << "Continuing into forming kernel segments\n";
        }
    }*/



}
