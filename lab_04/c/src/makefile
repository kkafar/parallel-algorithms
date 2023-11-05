.PHONY: build cleanall clean clean-data

all: build

build: main.c
	mpic++ -fast -o main -std=c++11 main.cpp

# run: build
# 	mpiexec -machinefile ./allnodes -np 2 ./main 1000000

cleanall: clean clean-data
	@echo "Cleaning all build artifacts & output files"

clean:
	rm -vfr main

clean-data:
	rm -vfr data
