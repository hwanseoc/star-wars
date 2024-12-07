
all:
	nvcc src/cuda.cu -std=c++20 -I./src -I./glm -O3 -arch=sm_86 -w --expt-relaxed-constexpr -o cuda_main

clean:
	rm cuda_main
	rm output.ppm
