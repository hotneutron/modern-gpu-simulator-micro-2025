GENCODE_SM90A := -gencode=arch=compute_90a,code=\"sm_90a,compute_90a\"

CUOPTS = $(GENCODE_SM90A) -std=c++17

CC := nvcc

SRC = ublkred_type_probe.cu

EXE = ublkred_type_probe

release:
	$(CC) $(CUOPTS) $(SRC) -o $(EXE)

clean:
	rm -f *.o; rm -f $(EXE)

run:
	./$(EXE)
