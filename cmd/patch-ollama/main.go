// Command patch-ollama prints the integration guide and optionally
// applies the patch to a local Ollama checkout.
package main

import (
	"fmt"
	"os"

	"github.com/databloom/ollama-kv-cache-tiering/kvcache"
)

func main() {
	if len(os.Args) > 1 && os.Args[1] == "--help" {
		fmt.Println("Usage: patch-ollama [--guide]")
		fmt.Println()
		fmt.Println("  --guide    Print the integration guide")
		fmt.Println()
		fmt.Println("To apply the patch to an Ollama checkout:")
		fmt.Println("  cd /path/to/ollama")
		fmt.Println("  git apply /path/to/ollama-kv-cache-tiering/patches/ollama-tiered-kvcache.patch")
		fmt.Println("  cp -r /path/to/ollama-kv-cache-tiering/diskstore .")
		fmt.Println("  go build .")
		os.Exit(0)
	}

	kvcache.PrintIntegrationGuide()
}
