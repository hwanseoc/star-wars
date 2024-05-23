
all:
	g++ main.cpp lodepng/lodepng.cpp -I. -I./lodepng -o main

clean:
	rm main
	rm output.png
