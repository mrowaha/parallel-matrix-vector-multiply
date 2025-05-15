CC = gcc
MPICC = mpicc

CFLAGS = -I.
MPIFLAGS = -I.

EXEC_SERIAL = serial
EXEC_MPI = parallel
ALL_EXEC = $(EXEC_SERIAL) $(EXEC_MPI)
ALL_EXEC = $(EXEC_MPI)

ifdef debug
    CFLAGS += -DDEBUG
    MPIFLAGS += -DDEBUG
endif

all: $(ALL_EXEC)

serial: serial-mvm.c config.h
	$(CC) $(CFLAGS) -o $@ $^

parallel: parallel.c config.h
	$(MPICC) $(MPIFLAGS) -o $@ $^ -lm

# Clean rule
clean:
	rm -f $(ALL_EXEC) *.txt
