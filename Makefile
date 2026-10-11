CXX ?= g++
ARCH ?= -march=native
OPTS ?= -O3 -ffast-math -funroll-loops
WARN ?= -Wall -Wextra
INCLUDES := -Iinclude
THREADS := -fopenmp
STD := -std=c++17

BUILD := build
BIN := $(BUILD)/bench
INFO := $(BUILD)/gguf-info
RUN := $(BUILD)/molot-run
CONV := $(BUILD)/gguf2molot
CHAT := $(BUILD)/molot-chat

all: $(BIN) $(INFO) $(RUN) $(CONV) $(CHAT)

$(BIN): bench/bench.cc $(wildcard include/molot/*.h) Makefile
	@mkdir -p $(BUILD)
	$(CXX) $(STD) $(OPTS) $(ARCH) $(WARN) $(INCLUDES) $(THREADS) $< -o $@

$(INFO): tools/gguf_info.cc $(wildcard include/molot/*.h) Makefile
	@mkdir -p $(BUILD)
	$(CXX) $(STD) $(OPTS) $(ARCH) $(WARN) $(INCLUDES) $< -o $@

$(RUN): tools/molot_run.cc $(wildcard include/molot/*.h) Makefile
	@mkdir -p $(BUILD)
	$(CXX) $(STD) $(OPTS) $(ARCH) $(WARN) $(INCLUDES) $(THREADS) $< -o $@

$(CONV): tools/gguf2molot.cc $(wildcard include/molot/*.h) Makefile
	@mkdir -p $(BUILD)
	$(CXX) $(STD) $(OPTS) $(ARCH) $(WARN) $(INCLUDES) $(THREADS) $< -o $@

$(CHAT): tools/molot_chat.cc $(wildcard include/molot/*.h) Makefile
	@mkdir -p $(BUILD)
	$(CXX) $(STD) $(OPTS) $(ARCH) $(WARN) $(INCLUDES) $(THREADS) $< -o $@

run: $(BIN)
	./$(BIN)

clean:
	rm -rf $(BUILD)

.PHONY: all run clean
