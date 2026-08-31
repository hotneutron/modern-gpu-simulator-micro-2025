CUOPTS = -arch=sm_90a -std=c++17 -O3

CC := nvcc
SRC = mbarrier_release_probe.cu
EXE = mbarrier_release_probe

release:
	$(CC) $(CUOPTS) $(SRC) -o $(EXE) -lcudart -lcuda

clean:
	rm -f *.o $(EXE)

run: release
	./$(EXE) --case count_many
	./$(EXE) --case count_alias
	./$(EXE) --case tma_many
	./$(EXE) --case tma_alias
