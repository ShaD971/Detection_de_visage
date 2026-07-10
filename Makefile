.PHONY: all build clean

all: build

build:
	cmake -S . -B build
	cmake --build build --config Release

clean:
	cmake --build build --target clean
