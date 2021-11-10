#include <stdlib.h>
#include <fstream>
#include <iostream>
//#include <boost/filesystem.hpp>

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
{BE
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

	// Demo Code -- read an mp4 video w/ known name
	// TODO: be able to read any name
	// TODO: read in avi files
	cv::VideoCapture video("/home/root/video.mp4");
	if(!video.open("/home/root/video.mp4")){
		std::cout << "[ERROR] -- video file unopenable" << std::endl;
		return EXIT_FAILURE;
	}

	// Demo Code -- read a single frame for round trip example
	// TODO: be able to read the entire video
	cv::Mat frame;
	video >> frame;
	if(frame.empty()){
		std::cout << "[ERROR] -- video file generated blank frame" << std::endl;
	}
	if(frame.rows != 1080 || frame.cols != 1920){
		std::cout << "[ERROR] -- bad dimensions from video frame" << std::endl;
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
	cl::Program::Binaries bins;
	bins.push_back({buf,nb});
	devices.resize(1);
	cl::Program program(context, devices, bins);

	// This call will get the kernel object from program. A kernel is an
	// OpenCL function that is executed on the FPGA.
	cl::Kernel krnl_srcnn(program,"cnn_top");

	// These commands will allocate memory on the Device. The cl::Buffer objects can
	// be used to reference the memory locations on the device.
	cl::Buffer buffer_I (context, CL_MEM_READ_ONLY, 1080*1920*4);
	cl::Buffer buffer_W1(context, CL_MEM_READ_ONLY, 1*2*9*9*4);
	cl::Buffer buffer_W2(context, CL_MEM_READ_ONLY, 2*2*5*5*4);
	cl::Buffer buffer_W3(context, CL_MEM_READ_ONLY, 2*1*5*5*4);
	cl::Buffer buffer_O1(context, CL_MEM_READ_WRITE, 1080*1920*4);
	cl::Buffer buffer_O2(context, CL_MEM_READ_WRITE, 1080*1920*4);
	cl::Buffer buffer_O3(context, CL_MEM_WRITE_ONLY, 1080*1920*4);

	//set the kernel Arguments
	int narg=0;
	krnl_srcnn.setArg(narg++,buffer_I );
	krnl_srcnn.setArg(narg++,buffer_W1);
	krnl_srcnn.setArg(narg++,buffer_W2);
	krnl_srcnn.setArg(narg++,buffer_W3);
	krnl_srcnn.setArg(narg++,buffer_O1);
	krnl_srcnn.setArg(narg++,buffer_O2);
	krnl_srcnn.setArg(narg++,buffer_O3);

	//We then need to map our OpenCL buffers to get the pointers
	float *ptr_I = (float *) q.enqueueMapBuffer (buffer_I , CL_TRUE, CL_MAP_WRITE, 0, 1080*1920*4);
	float *ptr_W1= (float *) q.enqueueMapBuffer (buffer_W1, CL_TRUE, CL_MAP_WRITE, 0, 1*2*9*9*4);
	float *ptr_W2= (float *) q.enqueueMapBuffer (buffer_W2, CL_TRUE, CL_MAP_WRITE, 0, 2*2*5*5*4);
	float *ptr_W3= (float *) q.enqueueMapBuffer (buffer_W3, CL_TRUE, CL_MAP_WRITE, 0, 2*1*5*5*4);
	float *ptr_O1= (float *) q.enqueueMapBuffer (buffer_O1, CL_TRUE,  CL_MAP_READ, 0, 1080*1920*4);
	float *ptr_O2= (float *) q.enqueueMapBuffer (buffer_O2, CL_TRUE,  CL_MAP_READ, 0, 1080*1920*4);
	float *ptr_O3= (float *) q.enqueueMapBuffer (buffer_O3, CL_TRUE,  CL_MAP_READ, 0, 1080*1920*4);

	//setting input data
	for(int row = 0; row<1080; row++){
		for(int col = 0; col<1920; col++){
			ptr_I[row*1920 + col] = frame.at<float>(row, col);
		}
	}

	//setting weights data
	// DEMO: using dummy data
	// TODO: don't use dummy data
	for(int i = 0; i<1*2*9*9; i++){
		ptr_W1[i] = ((100.0) * ((float)rand() / RAND_MAX));
	}
	for(int i = 0; i<2*2*5*5; i++){
		ptr_W2[i] = ((100.0) * ((float)rand() / RAND_MAX));
	}
	for(int i = 0; i<2*1*5*5; i++){
		ptr_W3[i] = ((100.0) * ((float)rand() / RAND_MAX));
	}


	// Data will be migrated to kernel space
	q.enqueueMigrateMemObjects({buffer_I,buffer_W1,buffer_W2,buffer_W3,buffer_O1,buffer_O2},0/* 0 means from host*/);

	//Launch the Kernel
	std::cout << " ========== START KERNEL ========== " << std::endl;
	q.enqueueTask(krnl_srcnn);
	std::cout << " ========== FINISH KERNEL ========== " << std::endl;

	std::cout << " ========== START WRITEBACK ========== " << std::endl;

	// The result of the previous kernel execution will need to be retrieved in
	// order to view the results. This call will transfer the data from FPGA to
	// source_results vector
	q.enqueueMigrateMemObjects({buffer_O1,buffer_O2,buffer_O3},CL_MIGRATE_MEM_OBJECT_HOST);

	//write back data
	for(int row = 0; row<1080; row++){
		for(int col = 0; col<1920; col++){
			frame.at<float>(row, col) = ptr_O3[row*1920 + col];
		}
	}

	std::cout << " ========== FINISH WRITEBACK ========== " << std::endl;

	q.finish();



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

	q.enqueueUnmapMemObject(buffer_I  , ptr_I );
	q.enqueueUnmapMemObject(buffer_W1 , ptr_W1);
	q.enqueueUnmapMemObject(buffer_W2 , ptr_W2);
	q.enqueueUnmapMemObject(buffer_W3 , ptr_W3);
	q.enqueueUnmapMemObject(buffer_O1 , ptr_O1);
	q.enqueueUnmapMemObject(buffer_O2 , ptr_O2);
	q.enqueueUnmapMemObject(buffer_O3 , ptr_O3);
	q.finish();

	// no correctness checking in demo code
	/*
	std::cout << "TEST " << (match ? "FAILED" : "PASSED") << std::endl;
	return (match ? EXIT_FAILURE :  EXIT_SUCCESS);
	*/

	cv::VideoWriter writer;
	int codec = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
	writer.open("/home/root/video_out.avi", codec, 30.0, frame.size(), frame.type()==CV_8UC3);
	if(!writer.isOpened()){
		std::cerr << "couldnt open writer" << std::endl;
		return 1;
	}

	//this should write a single framed video out to the file system
	writer.write(frame);

	std::cout << " ========== END OF DEMO ========== " << std::endl;
	return 0;



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
