.PHONY: test guide patch build-ollama clean

# Run tests for the diskstore package
test:
	go test ./diskstore/ -v -count=1

# Print the integration guide
guide:
	go run ./cmd/patch-ollama/

# Apply patch to a local Ollama checkout
# Usage: make patch OLLAMA_DIR=/path/to/ollama
OLLAMA_DIR ?= ../ollama
patch:
	@echo "=== Applying tiered KV cache patch to $(OLLAMA_DIR) ==="
	cp -r diskstore $(OLLAMA_DIR)/diskstore
	cd $(OLLAMA_DIR) && git apply $(CURDIR)/patches/ollama-tiered-kvcache.patch
	@echo "=== Patch applied. Build Ollama with: cd $(OLLAMA_DIR) && go build . ==="

# Build patched Ollama (assumes OLLAMA_DIR is already patched)
build-ollama:
	cd $(OLLAMA_DIR) && go generate ./... && go build .

# Clean test artifacts
clean:
	rm -rf /tmp/ollama-kv-cache /tmp/ollama-context-cache
