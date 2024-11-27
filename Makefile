
# g++ src/main.cpp lodepng/lodepng.cpp -std=c++20 -I./src -I./lodepng -I./glm -g -ffast-math -Wall -Wunused -Wshadow=local -Wdouble-promotion -o main
all:
# g++ src/main.cpp lodepng/lodepng.cpp -std=c++20 -I./src -I./lodepng -I./glm -O3 -ffast-math -Wall -Wunused -Wshadow=local -Wdouble-promotion -o main
	nvcc src/cuda.cu -std=c++20 -I./src -O3 -arch=sm_86 -w -o cuda_main

clean:
	rm cuda_main
	rm output.ppm
