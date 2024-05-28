IDIR = include
CC = g++
CFLAGS = -I$(IDIR) -I/usr/include/eigen3 $(shell pkg-config --cflags opencv4)
LDFLAGS = $(shell pkg-config --libs opencv4)

ODIR = obj
BDIR = bin

_DEPS = computes.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

_SRC = image_retrieval.cpp computes.cpp
SRC = $(patsubst %,src/%,$(_SRC))

_OBJ = $(patsubst %.cpp,%.o,$(_SRC))
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: src/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BDIR)/match: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ $(BDIR)/match



