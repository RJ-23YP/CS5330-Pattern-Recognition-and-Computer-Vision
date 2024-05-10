IDIR = include
CC = g++
CFLAGS = -I$(IDIR) $(shell pkg-config --cflags opencv4)
LDFLAGS = $(shell pkg-config --libs opencv4)

ODIR = obj
BDIR = bin

# Updated list of header files
_DEPS = faceDetect.h filters.h
DEPS = $(patsubst %,$(IDIR)/%,$(_DEPS))

# Updated list of source files
_SRC = vidDisplay.cpp filters.cpp faceDetect.cpp
SRC = $(patsubst %,src/%,$(_SRC))

# Updated list of object files
_OBJ = $(patsubst %.cpp,%.o,$(_SRC))
OBJ = $(patsubst %,$(ODIR)/%,$(_OBJ))

$(ODIR)/%.o: src/%.cpp $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

$(BDIR)/vid: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) $(LDFLAGS)

.PHONY: clean

clean:
	rm -f $(ODIR)/*.o *~ core $(INCDIR)/*~ $(BDIR)/vid


